import cv2
from matplotlib import pyplot as plt

def display(im_path: str):
    '''Display the image using matplotlib.

    Use matplotlib to load and display an image. Good for viewing inline in notebooks.

    Args:
        im_path (str): path to the image to be displayed
    
    Remarks:
        https://stackoverflow.com/questions/28816046/
    '''
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()


def conditional_save(image, save_to: str = None):
    '''Save an image to disk.

    Args:
        image (cv2 image): image to save to disk
        save_to (str): path to save the image to. does not save if equals None. default=None
    '''
    if save_to:
        cv2.imwrite(save_to, image)


def get_conditional_path(filename: str, folder: str = None) -> str:
    '''Join the filename and the folder if possible

    Args:
        filename (str): name of the file
        folder (str): folder to join the file on.
    
    Returns:
        str: path with folder and filename joined, if folder=None returns None
    '''
    return os.path.join(folder, filename) if folder else None

# Pre processing

def get_name(file_path, ext_size=4):
    last = file_path.replace('\\', '/').split('/')[-1]
    return last[:-ext_size] if ext_size > 0 else last

import pdf2image
import glob
import os
from unidecode import unidecode

def convert_pdfs(input_files: 'list[str]' = [], output_folder: str = './tmp', verbose: bool = False):
    '''Convert multiple PDF files into PNG images.

    Load PDFs and use pdf2image to convert each page into a different png image.

    Args:
        input_files (list): list of the paths to the PDF files to convert to PNG
        output_folder (str): destination folder to save the converted images
        verbose (bool): print extra information to console?
    
    Remarks:
        A directory is created inside the output folder for each input PDF,
        and the images correspondig to those PDF's pages are placed inside of
        said folder.

    '''

    for file_path in input_files:
        name = get_name(file_path)
        if verbose: print('processing file:', name)
        
        output = os.path.join(output_folder, name)
        os.makedirs(output, exist_ok=True)
        
        pdf2image.convert_from_path(file_path, output_folder=output, output_file='page', poppler_path='C:/Misc/poppler-21.09.0/Library/bin', fmt='png')
        
        if verbose:
            out_files = [ unidecode(get_name(f)) for f in glob.glob(f'{output}/*.png') ]
            print(f'\tconverted {len(out_files)} pages:', ','.join(out_files))


def load_image(path: str):
    image = cv2.imread(path)
    if image is None or image.size == 0:
        raise ValueError(f'failed to read image at "{path}, does it exist?"')
    return image

# Processing

from PIL import Image
import pytesseract

def run_ocr(image_path: str, output_path: str = None, temp_path: str = None, remove_spaces: bool = True, remove_hyphenation: bool = True, verbose: bool = False) -> 'tuple[str, float]':
    '''Detect portuguese text from an image using pytesseract.

    Load an image from a path and run it through pytesseract to detect text.

    Args:
        image_path (str): path of input image
        output_path (str): path to write text output to, does not save if equals None. default=None
        temp_folder (str): folder to save the image of the tesseract detected blocks, does not save if equals None. default=None
        remove_spaces (bool): flag to remove extra spaces in post-processing. default=True
        remove_hyphenation (bool): flag to remove hyphenation, joining words in post-processing. default=True
        verbose (bool): write extra information to console?

    Returns:
        tuple(str, float): text detected and mean confidence score
    '''
    img = Image.open(image_path)
    if verbose:
        print(f'read image from "{image_path}"')
    
    data = pytesseract.image_to_data(img, lang='por', output_type=pytesseract.Output.DATAFRAME)
    conf = data[data['conf'] > -1]['conf'].mean()
    data.loc[data['level'] < 5, 'text'] = '\n'
    data.loc[data['level'] == 5, 'text'] = data.loc[data['level'] == 5, 'text'].fillna('') + ' '
    data['page_block_par_num'] = (data['page_num'].astype(str).str.rjust(3,'0') +
                                   data['block_num'].astype(str).str.rjust(3,'0') +
                                   data['par_num'].astype(str).str.rjust(3,'0')).astype(int)
    lines = data.groupby('page_block_par_num')['text'].sum().str.strip() + '\n'
    result = lines.sum().strip()
    if verbose:
        print(f'detected {len(result)} characters in image')
    
    blocks = data[data['level'] == 3]
    cvImg = cv2.imread(image_path)
    for _, block in blocks.iterrows():
        cv2.rectangle(cvImg, (block['left'], block['top']), (block['left'] + block['width'], block['top'] + block['height']), (0, 255, 0), 2)
    conditional_save(cvImg, temp_path)

    if remove_spaces:
        result = remove_extra_spaces(result)
        if verbose: print(f'after space removal, got {len(result)} characters')
    
    if remove_hyphenation:
        result = treat_hyphenation(result)
        if verbose: print(f'after hyphenation removal, got {len(result)} characters')

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            if verbose: print(f'writing result to "{output_path}"')
            f.write(result)
    
    return result, conf

def run_ocr_on_columns(columns_path: 'list[str]', temp_folder: str, output_path: str, verbose: bool = False) -> 'tuple[str, float]':
    '''Detect text from multiple images and append them together

    Args:
        columns_path (list[str]): list of paths to the images to process
        temp_folder (str): path to a directory to write the text files for each image
        output_path (str): path to a text file to write the final output

    Returns:
        tuple(str, float): All of the detected texts and mean confidence score
    '''
    avg_conf = 0
    for i in range(len(columns_path)):
        out = os.path.join(temp_folder, f'{i}.txt')
        _, conf = run_ocr(columns_path[i], out, verbose=verbose)
        avg_conf += conf
    avg_conf /= len(columns_path)

    result = []
    for i in range(len(columns_path)):
        text_file = os.path.join(temp_folder, f'{i}.txt')
        with open(text_file, encoding='utf-8') as f:
            result.append(f.read())

    result = '\n\n'.join(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result, avg_conf

# Post processing

import re
def remove_extra_spaces(text: str) -> str:
    '''Remove double spaces and extra line feeds.

    Args:
        text (str): text to remove spaces from
    
    Returns:
        str: text with no double spaces or 3+ line feeds
    '''
    text = re.sub('\n{2,}', '\n\n', text)
    text = re.sub(' +', ' ', text)
    return text

def treat_hyphenation(text: str) -> str:
    '''Remove hyphenation from text.

    Args:
        text (str): text to remove hyphenation from
    
    Returns:
        str: text without hyphenation
    '''
    text = re.sub('- *\n *', '', text)
    return text

import pdf2image
import glob
import os

def convert_pdfs(input_files=[], output_folder='./tmp', verbose=False, is_dev=True):
    def get_name(file_path):
        return file_path.replace('\\', '/').split('/')[-1][:-4]

    for file_path in input_files:
        name = get_name(file_path)
        if verbose: print('processing file:', name)
        
        output = os.path.join(output_folder, name)
        os.makedirs(output, exist_ok=True)
        
        poppler_path = 'C:/Misc/poppler-21.09.0/Library/bin' if is_dev else None
        pdf2image.convert_from_path(file_path, output_folder=output, output_file='page', poppler_path=poppler_path, fmt='png')
        
        if verbose:
            out_files = [ get_name(f) for f in glob.glob(f'{output}/*.png') ]
            print(f'\tconverted {len(out_files)} pages:', ','.join(out_files))

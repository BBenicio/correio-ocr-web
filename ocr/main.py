import glob
import os
import cv2
import argparse
from unidecode import unidecode
from tqdm import tqdm

from process_pdfs import convert_pdfs
from image_prep import deskew, prepare_image
from image_processing import extract_page
from mhs_layout_analisys import segment
import utils

parser = argparse.ArgumentParser(description='Reconhece jornais históricos Correio da Lavoura.')
parser.add_argument('--production', '-p', action='store_true', help='flag if is running in productive environment')
parser.add_argument('--pdf', action='store_true', help='flag if the input is one or more pdf files.')
parser.add_argument('--mhs', action='store_true', help='flag to use mhs segmentation before running tesseract.')
parser.add_argument('--verbose', '-v', action='store_true', help='print information messages to console.')
parser.add_argument('--edition', '-e', type=str, help='only run on the specified edition name')
parser.add_argument('--output', '-o', type=str, help='directory to store the output in')
parser.add_argument('input', nargs='*', type=str, help='input files. if flag --pdf is used, files must be PDFs, otherwise PNGs are expected.')
args = parser.parse_args()

PROCESS_PDFS = args.pdf
DO_OCR = True
OCR_BASE = DO_OCR and False
OCR_GRAY = DO_OCR and False
OCR_PROCESSED = DO_OCR and True
DO_MHS = args.mhs
VERBOSE = args.verbose

def log(msg):
    if VERBOSE:
        print(msg)

input_files = []
for input_file in args.input:
    input_files.extend(glob.glob(input_file))

os.makedirs('./input/processed', exist_ok=True)
if PROCESS_PDFS:
    log('converting PDFs into PNGs')
    convert_pdfs(input_files, './input/processed', VERBOSE, is_dev=not args.production)

all_files = []

editions = [f'./input/processed/{args.edition}'] if args.edition else glob.glob('./input/processed/*')
for ed in editions:
    ed_name = unidecode(utils.get_name(ed, 0).lower())
    pages = glob.glob(f'{ed}/*.png')
    for page in pages:
        page_name = utils.get_name(page)
        all_files.append((ed_name, page_name, page))

for ed_name, page_name, page in tqdm(all_files):
    log(f'...in page "{page_name}" from "{ed_name}"')
    image = utils.load_image(page)

    output_path = os.path.join(args.output, page_name) if args.output else f'./output/{ed_name}/{page_name}'
    
    os.makedirs(f'./temp/{ed_name}/{page_name}', exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    log('cropping image')
    image, _ = extract_page(image, f'./temp/{ed_name}/{page_name}/', f'./temp/{ed_name}/{page_name}/cropped.png')

    log('preparing image')
    image = prepare_image(image, f'./temp/{ed_name}/{page_name}/prepared.png', f'./temp/{ed_name}/{page_name}', verbose=VERBOSE)


    if DO_MHS:
        image, _, _ = segment(image, f'./temp/{ed_name}/{page_name}/')
        image = deskew(image)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        utils.conditional_save(image, f'./temp/{ed_name}/{page_name}/rotated_after_mhs.png')
        utils.conditional_save(image, f'./temp/{ed_name}/{page_name}.png')
    else:
        image = deskew(image)
        utils.conditional_save(image, f'./temp/{ed_name}/{page_name}.png')

    if OCR_BASE:
        log('running OCR on the unprocessed page')
        utils.run_ocr(page, os.path.join(output_path, 'base.txt'), f'./temp/{ed_name}/{page_name}/tess_unproc.png', verbose=VERBOSE)

    if OCR_GRAY:
        log('running OCR on the grayscale page')
        utils.run_ocr(f'./temp/{ed_name}/{page_name}/grayscale.png', os.path.join(output_path, 'gray.txt'), f'./temp/{ed_name}/{page_name}/tess_gray.png', verbose=VERBOSE)

    if OCR_PROCESSED:
        log('running OCR on the processed page')
        utils.run_ocr(f'./temp/{ed_name}/{page_name}.png', os.path.join(output_path, 'proc.txt'), f'./temp/{ed_name}/{page_name}/tess_proc.png', verbose=VERBOSE)
    

    log(f'DONE with page "{page_name}" from "{ed_name}"')

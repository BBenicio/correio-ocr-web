import os
import re
import glob
import pandas as pd
from unidecode import unidecode

import utils

def count_characters(output_path='quality.tsv'):
    all_files = []

    editions = glob.glob('./output/*')
    for ed in editions:
        ed_name = utils.get_name(ed, 0).lower()
        pages = glob.glob(f'{ed}/*')
        for page in pages:
            page_name = utils.get_name(page, 0)
            all_files.append((ed_name, page_name, page))
        
    df = pd.DataFrame(columns=['edition', 'page', 'base', 'grayscale', 'processed'])

    for ed_name, page_name, page in all_files:
        base_path = os.path.join(page, 'base.txt')
        gray_path = os.path.join(page, 'gray.txt')
        proc_path = os.path.join(page, 'processed.txt')
        results = { 'edition': ed_name, 'page': page_name, 'base': 0, 'grayscale': 0, 'processed': 0 }
        
        with open(base_path, 'r', encoding='utf-8') as f:
            text = unidecode(f.read())
            text = re.sub(r'[^\w]+', '', text)
            results['base'] = len(text)
        
        with open(gray_path, 'r', encoding='utf-8') as f:
            text = unidecode(f.read())
            text = re.sub(r'[^\w]+', '', text)
            results['grayscale'] = len(text)

        with open(proc_path, 'r', encoding='utf-8') as f:
            text = unidecode(f.read())
            text = re.sub(r'[^\w]+', '', text)
            results['processed'] = len(text)
        
        df = df.append(results, ignore_index=True)

    df.to_csv(output_path, index=False, sep='\t')
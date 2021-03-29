from os.path import join
from os import listdir
import pdb

import shutil

from database import read_slice_info
from setup import DATA_DIR
from src.utils.io import create_results_dir


data_dir = join(DATA_DIR, 'dataset')
create_results_dir(data_dir, subdirs=['ihc','mri','nissl'])

shutil.copy('slice_info_ihc.csv', join(data_dir, 'ihc', 'slice_separation.csv'))
shutil.copy('slice_info_nissl.csv', join(data_dir, 'nissl', 'slice_separation.csv'))

slice_dict = read_slice_info(key='slice_number')
for stain in ['NISSL', 'IHC']:
    files = listdir(join(DATA_DIR, stain.lower(), 'images_orig'))
    for f in files:
        slice_num = str(int(f.split('_')[1].split('.')[0]))
        pdb.set_trace()
        outf = slice_dict[stain][slice_num]['filename']
        shutil.move(join(DATA_DIR, stain.lower(), 'images_orig', f),join(DATA_DIR, stain.lower(), 'images', outf))
        shutil.move(join(DATA_DIR, stain.lower(), 'masks_orig', f),join(DATA_DIR, stain.lower(), 'masks', outf))

shutil.rmtree(join(DATA_DIR, stain.lower(), 'images_orig'))
shutil.rmtree(join(DATA_DIR, stain.lower(), 'masks_orig'))
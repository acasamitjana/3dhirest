# python imports
import os
from os.path import join

# external libraries imports

# project imports
from setup import DATA_DIR
from database import data_loader as DL_Allen

# It is important that all files contain the same structure: prefix + '_' + slice_num + '.' + extension

DATA_LOADER_DICT = {
    'Allen': DL_Allen,
}

SLICE_PREFIX = 'slice'
BASE_DIR = join(DATA_DIR, 'dataset')


Allen_MRI = {
    'DATABASE': DL_Allen,
    'BASE_DIR': BASE_DIR,
    'DATASET_SHEET': join(BASE_DIR, 'mri/slice_separation.csv'),
    'SLICE_PREFIX': 'slice_',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'Allen_MRI'
}

Allen_IHC = {
    'DATABASE': DL_Allen,
    'BASE_DIR': BASE_DIR,
    'DATASET_SHEET': join(BASE_DIR, 'ihc/slice_separation.csv'),
    'SLICE_PREFIX': 'slice_',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'Allen_ihc'
}

Allen_NISSL = {
    'DATABASE': DL_Allen,
    'BASE_DIR': BASE_DIR,
    'DATASET_SHEET': join(BASE_DIR, 'nissl/slice_separation.csv'),
    'SLICE_PREFIX': 'slice_',
    'IMAGE_EXTENSION': '.png',
    'NAME': 'Allen_nissl'
}




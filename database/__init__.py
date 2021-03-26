from collections import OrderedDict
from os.path import join
import csv

from setup import DATA_DIR


def read_slice_info(file='slice_separation.csv', stain=['IHC', 'NISSL'], key='filename', data_dir=join(DATA_DIR, 'dataset')):
    slice_num_dict = OrderedDict({k: {} for k in stain})
    for k in slice_num_dict.keys():
        mapping_file = join(data_dir, k.lower(), file)
        with open(mapping_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for it_row, row in enumerate(csvreader):
                slice_num_dict[k][row[key]] = row
    return slice_num_dict
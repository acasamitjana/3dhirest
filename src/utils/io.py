from datetime import datetime, date
from os.path import join, exists
from os import makedirs
import gc

import torch
import numpy as np

def worker_init_fn(wid):
    np.random.seed(np.mod(torch.utils.data.get_worker_info().seed, 2**32-1))

def create_results_dir(results_dir, subdirs=None):
    if subdirs is None:
        subdirs = ['checkpoints', 'results']
    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))

class DebugWriter(object):

    def __init__(self, debug_flag, filename = None, attach = False):
        self.filename = filename
        self.debug_flag = debug_flag
        if filename is not None:
            date_start = date.today().strftime("%d/%m/%Y")
            time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if not attach:
                with open(self.filename, 'w') as writeFile:
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')
            else:
                with open(self.filename, 'a') as writeFile:
                    for i in range(4):
                        writeFile.write('\n')
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')

    def write(self, to_write):
        if self.debug_flag:
            if self.filename is None:
                print(to_write, end=' ')
            else:
                with open(self.filename, 'a') as writeFile:
                    writeFile.write(to_write)

class ResultsWriter(object):

    def __init__(self, filename = None, attach = False):
        self.filename = filename
        if filename is not None:
            date_start = date.today().strftime("%d/%m/%Y")
            time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if not attach:
                with open(self.filename, 'w') as writeFile:
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')
            else:
                with open(self.filename, 'a') as writeFile:
                    for i in range(4):
                        writeFile.write('\n')
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')

    def write(self, to_write):
        if self.filename is None:
            print(to_write, end=' ')
        else:
            with open(self.filename, 'a') as writeFile:
                writeFile.write(to_write)

class ExperimentWriter(object):
    def __init__(self, filename = None, attach = False):
        self.filename = filename
        date_start = date.today().strftime("%d/%m/%Y")
        time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if filename is not None:
            method = 'a' if attach else 'w'
            with open(filename, method) as writeFile:
                writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                writeFile.write('\n')

    def write(self, to_write):
        if self.filename is None:
            print(to_write, end=' ')
        else:
            with open(self.filename, 'a') as writeFile:
                writeFile.write(to_write)


def check_gc_torch():
    it_obj = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype)
                it_obj += 1
        except:
            pass

    print('Total number of tensors ' + str(it_obj))

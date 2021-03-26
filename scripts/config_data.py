from os.path import join, exists
from os import makedirs

import numpy as np

from setup import RESULTS_DIR

## Results path
OBSERVATIONS_DIR_REGNET = join(RESULTS_DIR, 'RegNet')
OBSERVATIONS_DIR_NR = join(RESULTS_DIR, 'NiftyReg')
ALGORITHM_DIR = join(RESULTS_DIR, 'ST')

if not exists(OBSERVATIONS_DIR_REGNET):
    makedirs(OBSERVATIONS_DIR_REGNET)
if not exists(OBSERVATIONS_DIR_NR):
    makedirs(OBSERVATIONS_DIR_NR)
if not exists(ALGORITHM_DIR):
    makedirs(ALGORITHM_DIR)

## Linear MRI affine matrix
MRI_AFFINE = np.asarray([
    [0, -0.5, 0, 55.25],
    [0, 0, -0.5, 0],
    [-0.5, 0, 0, 79.25],
    [0, 0, 0, 1]
])

# Data characteristics
HISTO_RES = 0.032 # Initial histo resolution
HISTO_THICKNESS = 0.05 # Initial histo section thickness
INIT_RES = 0.5 # Initial resolution of linearly aligned volumes
NSLICES = 641 # Number of slices in the tree
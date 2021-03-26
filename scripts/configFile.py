import copy
from os.path import join, exists
from os import makedirs, environ

import numpy as np

from setup import RESULTS_DIR
from src.utils.image_transform import ScaleNormalization, NonLinearParams, RotationParams, CropParams
from database import databaseConfig

OBSERVATIONS_DIR_REGNET = join(RESULTS_DIR, 'RegNet')
OBSERVATIONS_DIR_NR = join(RESULTS_DIR, 'NiftyReg')
ALGORITHM_DIR = join(RESULTS_DIR, 'ST')

if not exists(OBSERVATIONS_DIR_REGNET):
    makedirs(OBSERVATIONS_DIR_REGNET)
if not exists(OBSERVATIONS_DIR_NR):
    makedirs(OBSERVATIONS_DIR_NR)
if not exists(ALGORITHM_DIR):
    makedirs(ALGORITHM_DIR)

IMAGE_SHAPE = (448, 320)

MRI_AFFINE = np.asarray([
    [0, -0.5, 0, 55.25],
    [0, 0, -0.5, 0],
    [-0.5, 0, 0, 79.25],
    [0, 0, 0, 1]
])

HISTO_RES = 0.032
HISTO_THICKNESS = 0.05
BLOCK_RES = 0.25
INIT_RES = 0.5
NSLICES = 641
CROP_XY = [93 * BLOCK_RES, 79 * BLOCK_RES]

HISTO_AFFINE = np.zeros((4, 4))
HISTO_AFFINE[0, 1] = -BLOCK_RES
HISTO_AFFINE[2, 0] = -BLOCK_RES
HISTO_AFFINE[1, 2] = -HISTO_THICKNESS
HISTO_AFFINE[0, 3] = -93 * BLOCK_RES
HISTO_AFFINE[2, 3] = -79 * BLOCK_RES


CONFIG_DATA = {
    'DB_CONFIG': databaseConfig.Allen_MRI,
    'TRANSFORM': [CropParams(crop_shape=IMAGE_SHAPE)],
    'IMAGE_SHAPE': IMAGE_SHAPE,

    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0,1]),

    'REF_MODALITY': 'MRI',
    'FLO_MODALITY': 'MRI'
}

CONFIG_REGISTRATION = {
    'DB_CONFIG': databaseConfig.Allen_MRI,

    'TRANSFORM': [CropParams(crop_shape=IMAGE_SHAPE)],
    'IMAGE_SHAPE': IMAGE_SHAPE,

    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0, 1]),
    'ROTATION': RotationParams([0,0]),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9], lowres_strength=[3,8], distribution='uniform'),

    'ENC_NF': [16, 32, 32, 64, 64, 64],
    'DEC_NF': [64, 64, 64, 32, 32, 32, 32, 16, 16],
    'INT_STEPS': 7,

    'BATCH_SIZE': 2,
    'N_EPOCHS': 200,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,# Set to False if running to CPU
    'GPU_INDICES': [0],

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [9,9], 'kernel_type': 'mean'},'lambda': 1},
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 2, 'penalty': 'l2'}, 'lambda': 1},
    'LOSS_MAGNITUDE': {'name': 'Norm2', 'params': {}, 'lambda': 0.01},

    'UPSAMPLE_LEVELS': 8,
    'FIELD_TYPE': 'velocity',
}

CONFIG_INTRAMODAL = copy.copy(CONFIG_REGISTRATION)
CONFIG_INTRAMODAL['NEIGHBOR_DISTANCE'] = 4

CONFIG_INTERMODAL = copy.copy(CONFIG_REGISTRATION)

CONFIG_MRI = copy.copy(CONFIG_INTRAMODAL)
CONFIG_MRI['DB_CONFIG'] = databaseConfig.Allen_MRI
CONFIG_MRI['REF_MODALITY'] = 'MRI'
CONFIG_MRI['FLO_MODALITY'] = 'MRI'

CONFIG_NISSL = copy.copy(CONFIG_INTRAMODAL)
CONFIG_NISSL['DB_CONFIG'] = databaseConfig.Allen_NISSL
CONFIG_NISSL['REF_MODALITY'] = 'NISSL'
CONFIG_NISSL['FLO_MODALITY'] = 'NISSL'

CONFIG_IHC = copy.copy(CONFIG_INTRAMODAL)
CONFIG_IHC['DB_CONFIG'] = databaseConfig.Allen_IHC
CONFIG_IHC['REF_MODALITY'] = 'IHC'
CONFIG_IHC['FLO_MODALITY'] = 'IHC'

CONFIG_MRI_NISSL = copy.copy(CONFIG_INTERMODAL)
CONFIG_MRI_NISSL['DB_CONFIG'] = databaseConfig.Allen_NISSL
CONFIG_MRI_NISSL['REF_MODALITY'] = 'MRI'
CONFIG_MRI_NISSL['FLO_MODALITY'] = 'NISSL'

CONFIG_MRI_IHC = copy.copy(CONFIG_INTERMODAL)
CONFIG_MRI_IHC['DB_CONFIG'] = databaseConfig.Allen_IHC
CONFIG_MRI_IHC['REF_MODALITY'] = 'MRI'
CONFIG_MRI_IHC['FLO_MODALITY'] = 'IHC'

CONFIG_IHC_NISSL = copy.copy(CONFIG_INTERMODAL)
CONFIG_IHC_NISSL['DB_CONFIG'] = databaseConfig.Allen_IHC
CONFIG_IHC_NISSL['REF_MODALITY'] = 'IHC'
CONFIG_IHC_NISSL['FLO_MODALITY'] = 'NISSL'


CONFIG_DICT = {
    'BASE': CONFIG_REGISTRATION,
    'MRI': CONFIG_MRI,
    'IHC': CONFIG_IHC,
    'NISSL': CONFIG_NISSL,
    'MRI_IHC': CONFIG_MRI_IHC,
    'MRI_NISSL': CONFIG_MRI_NISSL,
    'IHC_NISSL': CONFIG_IHC_NISSL

}

for k, config in CONFIG_DICT.items():
    if config['LOSS_REGISTRATION']['name'] == 'NCC':
        loss_dir = config['LOSS_REGISTRATION']['name'] + str(config['LOSS_REGISTRATION']['params']['kernel_var'][0])
    else:
        loss_dir = config['LOSS_REGISTRATION']['name']
    loss_name = 'R' + str(config['LOSS_REGISTRATION']['lambda'])
    loss_name += '_S' + str(config['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
    loss_dir = join(loss_dir, loss_name)
    CONFIG_DICT[k]['RESULTS_DIR'] = join(RESULTS_DIR, loss_dir, 'DownFactor_' + str(config['UPSAMPLE_LEVELS']), k)


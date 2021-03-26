# py
from os.path import join, exists
import subprocess
import copy
from argparse import ArgumentParser

# libraries imports
import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# project imports
from setup import DATA_DIR, NIFTY_REG_DIR
from scripts import config_dev as configFile, config_data

from database.data_loader import DataLoader as DL_subject
from src.utils.image_utils import bilinear_interpolate, deform2D
from database import read_slice_info


data_path_dict = {
    'RegNet': config_data.OBSERVATIONS_DIR_REGNET,
    'NR': config_data.OBSERVATIONS_DIR_NR,
    'ST3_L1_RegNet': join(config_data.ALGORITHM_DIR, 'ST3_RegNet', 'l1', 'NN4'),
    'ST3_L2_RegNet': join(config_data.ALGORITHM_DIR, 'ST3_RegNet', 'l2', 'NN4'),
    'ST3_L1_NR': join(config_data.ALGORITHM_DIR, 'ST3_NR', 'l1', 'NN4'),
    'ST3_L2_NR': join(config_data.ALGORITHM_DIR, 'ST3_NR', 'l2', 'NN4')
}


TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
INIT_BASE_DIR = join(DATA_DIR, 'downloads')
BASE_DIR = join(DATA_DIR, 'dataset')

file = 'slice_separation.csv'
slice_num_dict = read_slice_info(file, stain=['MRI', 'IHC', 'NISSL'], key='filename')


HISTO_RES = config_data.HISTO_RES
HISTO_THICKNESS = config_data.HISTO_THICKNESS
BLOCK_RES = configFile.BLOCK_RES
HISTO_AFFINE = configFile.HISTO_AFFINE
image_shape = configFile.IMAGE_SHAPE

# Parameters
arg_parser = ArgumentParser(description='Computes the deformation between original images for different'
                                        'registration algorithms at the resolution of choice')
arg_parser.add_argument('--outres', type=float, default=BLOCK_RES, help='Resolution of the output volume')
arg_parser.add_argument('--reg_algorithm', type=str, default='RegNet', choices=list(data_path_dict.keys()),
                        help='Registration algorithm from which get the deformation fields')

arguments = arg_parser.parse_args()
OUTPUT_RES = arguments.outres
reg_algorithm = arguments.reg_algorithm


print('Deforming images ...')

data_loader_subject = DL_subject(configFile.CONFIG_MRI)
affine = copy.copy(HISTO_AFFINE)
nslices = len(data_loader_subject.subject_list[0].slice_list)
inplane_factor = BLOCK_RES / OUTPUT_RES
if inplane_factor != 1:
    final_shape = tuple([int(i * inplane_factor) for i in image_shape])
    inplane_factor = tuple([a / b for a, b in zip(final_shape, image_shape)])

    aux = np.asarray([(ipf - 1) / (2 * ipf) for ipf in inplane_factor] + [0])
    affine[:3, 0] = affine[:3, 0] / inplane_factor[0]
    affine[:3, 1] = affine[:3, 1] / inplane_factor[1]
    affine[:3, 3] = affine[:3, 3] - np.dot(HISTO_AFFINE[:3, :3], aux.T)
else:
    final_shape = image_shape


for stain in ['NISSL','IHC']:
    print('\n # ' + stain)

    if exists(join(data_path_dict[reg_algorithm], stain + '.flow.tree.nii.gz')):
        proxy = nib.load(join(data_path_dict[reg_algorithm], stain + '.flow.tree.nii.gz'))
    elif exists(join(data_path_dict[reg_algorithm], 'MRI_' + stain + '.flow.tree.nii.gz')):
        proxy = nib.load(join(data_path_dict[reg_algorithm], 'MRI_' + stain + '.flow.tree.nii.gz'))
    else:
        raise ValueError("[FLOW NOT FOUND] Please, compute the deformation fields for each block and group them "
                         "using visualization/group_blocks.py first")
    total_flow = np.asarray(proxy.dataobj)
    init_sbj = 0

    print('   ## Slice number:', end=' ', flush=True)
    for slice_filename, slice_info in slice_num_dict[stain].items():
        print(slice_info['tree_pos'], end=' ', flush=True)
        tree_pos = int(slice_info['tree_pos'])
        slice_num = int(slice_info['slice_number'])

        #### Compute the aggregate FIELD
        dummyFileNifti = 'images_deform.nii.gz'
        refFile = join(BASE_DIR, 'mri', 'images', slice_filename)
        affineFile = join(BASE_DIR, stain.lower(), 'affine', slice_filename[:-3] + 'aff')
        subprocess.call([TRANSFORMcmd, '-ref', refFile, '-disp', affineFile, dummyFileNifti], stdout=subprocess.DEVNULL)
        proxy_aff = nib.load(dummyFileNifti)
        affine_field = np.asarray(proxy_aff.dataobj)[:, :, 0, 0, :]
        affine_field = np.transpose(affine_field, axes=[2, 1, 0])

        YY, XX = np.meshgrid(np.arange(0, total_flow.shape[1]), np.arange(0, total_flow.shape[2]), indexing='ij')
        XX2 = XX + total_flow[0,..., tree_pos]
        YY2 = YY + total_flow[1,..., tree_pos]
        incx = bilinear_interpolate(affine_field[0], XX2, YY2)
        incy = bilinear_interpolate(affine_field[1], XX2, YY2)
        XX3 = XX2 + incx
        YY3 = YY2 + incy
        field_x = XX3 - XX
        field_y = YY3 - YY

        #### Read image and deform
        H_filepath = join(INIT_BASE_DIR, stain.lower(), 'images', slice_filename)
        H_mask_filepath = join(INIT_BASE_DIR, stain.lower(), 'masks', slice_filename)

        H_orig = cv2.imread(H_filepath)
        H = np.zeros_like(H_orig, dtype='float')
        for it_c in range(3):
            H[..., it_c] = gaussian_filter((H_orig[..., it_c] / 255).astype(np.double), sigma=5)

        M = cv2.imread(H_mask_filepath, flags=0)
        M = (M / np.max(M)) > 0.5
        M = (255 * M).astype('uint8')

        resized_shape = tuple([int(i * HISTO_RES / BLOCK_RES) for i in H_orig.shape])
        H = cv2.resize(H, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
        M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
        M = (M / np.max(M)) > 0.5

        H = (255 * H).astype('uint8')
        H = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)

        H_masked = np.zeros_like(H, dtype='uint8')
        for it_c in range(3):
            H_sl = H[..., it_c]
            H_sl[~M] = 0
            H_masked[..., it_c] = H_sl

        field = np.zeros((2,) + field_x.shape)
        field[0] = field_x
        field[1] = field_y

        if 'image_volume' not in locals():
            num_slices_tree = config_data.NSLICES
            image_volume = np.zeros(final_shape + (num_slices_tree, 3))

        if 'mask_volume' not in locals():
            num_slices_tree = config_data.NSLICES
            mask_volume = np.zeros(final_shape + (num_slices_tree,))

        mask = deform2D(M, field, mode='bilinear')
        mask = cv2.resize(mask, (final_shape[1], final_shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_volume[..., tree_pos] = mask
        for it_s in range(3):
            image = deform2D(H[..., it_s], field, mode='bilinear')
            image = cv2.resize(image, (final_shape[1], final_shape[0]), interpolation=cv2.INTER_LINEAR)
            image_volume[..., tree_pos, it_s] = image


    img = nib.Nifti1Image(image_volume, affine)
    nib.save(img, join(data_path_dict[reg_algorithm], stain + '_IMAGE_' + str(OUTPUT_RES) + '.tree.nii.gz'))

    img = nib.Nifti1Image(mask_volume, affine)
    nib.save(img, join(data_path_dict[reg_algorithm], stain + '_MASK_' + str(OUTPUT_RES) + '.tree.nii.gz'))

    del image_volume
    del mask_volume

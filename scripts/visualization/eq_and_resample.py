from os.path import join, dirname, basename
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression

from setup import DATA_DIR
from scripts import config_dev as configFile, config_data
from src.utils.image_utils import build_regular_space_volume, build_regular_space_volume_color

# Initial parameters
nslices = config_data.NSLICES
image_shape = configFile.IMAGE_SHAPE
mri_dir = join(DATA_DIR, 'dataset', 'mri')
input_z_pos = join(DATA_DIR, 'downloads', 'linear', 'z_pos.npy')
z_pos = 1 * np.squeeze(np.load(input_z_pos))

INIT_RES = config_data.INIT_RES
MRI_AFFINE = config_data.MRI_AFFINE
CROP_XY = configFile.CROP_XY
Trot = np.array([[0, 0, -1, -CROP_XY[0]], [-1, 0, 0, -CROP_XY[1]],  [0, -1, 0, 0], [0, 0, 0, 1]]) # brings MRI into the orientation of the stack
TrotInv = np.linalg.inv(Trot)

## Arguments
arg_parser = ArgumentParser(description='Simple equalization (linear regression) and resampling of a 3D'
                                        'histology reconstruction contrast (grayscale, color images')
arg_parser.add_argument('--zres', type=float, default=0.5, help='Resolution on the direction of the stack')
arg_parser.add_argument('--filepath', type=str, help='Filepath of \"*.tree\" file to equalize ')
arg_parser.add_argument('--mfilepath',  default=None,  help='Filepath in \"*.tree\" format used to mask the image file. '
                                                            'Optional: (str, None) ')

arguments = arg_parser.parse_args()
filepath = arguments.filepath
zres = arguments.zres
mfilepath = arguments.mfilepath

assert '.tree' in filepath
filename = basename(filepath).split('.tree')[0]

## Parameters of the equalization model
outlier_limits = (140, 170)
brightness_f = lambda R, G, B: 0.2126 * R + 0.7152 * G + 0.0722 * B

## Read images
print('Reading images ....')
proxy = nib.load(join(mri_dir, 'MRI_images.nii.gz'))
data_M = np.asarray(proxy.dataobj)

proxy = nib.load(filepath)
IMAGE_RES = np.linalg.norm(proxy.affine, 2, axis=0)[0]
data_H = np.asarray(proxy.dataobj)

if data_H.shape[-1] == 3:
    colored = True
else:
    colored = False

if mfilepath is not None:
    proxy = nib.load(mfilepath)
    mask_H = np.asarray(proxy.dataobj) > 0.5
    if colored:
        data_H[~mask_H, 0] = 0
        data_H[~mask_H, 1] = 0
        data_H[~mask_H, 2] = 0
    else:
        data_H[mask_H==0]=0
else:
    mask_H = data_H > 0

## Compute image statistics
print('Computing image statistics ...')
median_H = []
median_M = []
for it_s in range(nslices):
    H = brightness_f(data_H[..., it_s, 0], data_H[..., it_s, 1], data_H[..., it_s, 2]) if colored else data_H[..., it_s]
    M = data_M[..., it_s]

    H_M = mask_H[..., it_s]
    M_M = M > 0
    m = np.median(M[M_M])
    if m > outlier_limits[0] and m < outlier_limits[1] and np.sum(H_M) > 0:
        h = np.median(H[H_M])
        median_H.append(h)
        median_M.append(m)

## Linear regression of the median intensity of MRI slices and Histology sections
lr = LinearRegression()
lr.fit(np.asarray(median_M).reshape(-1,1), np.asarray(median_H).reshape(-1,1))
a = lr.coef_
b = lr.intercept_

print('Equalizing ...')
## Equalization (i.e. brightness and contrast) with the deviation from the previous linear regression
data_H_eq = np.zeros(image_shape + (nslices,3)) if colored else np.zeros(image_shape + (nslices,))
for it_s in range(nslices):
    H = brightness_f(data_H[..., it_s, 0], data_H[..., it_s, 1], data_H[..., it_s, 2]) if colored else data_H[..., it_s]
    M = data_M[..., it_s]

    H_M = mask_H[..., it_s]
    M_M = M > 0
    m = np.median(M[M_M])
    if np.sum(H_M) > 0:
        h = np.median(H[H_M])
        new_h = a * m + b
        if colored:
            new_H = new_h / h * data_H[..., it_s, :]
            data_H_eq[..., it_s,:] = np.clip(new_H, 0, 255)
        else:
            new_H = new_h/h*H
            data_H_eq[..., it_s] = np.clip(new_H, 0, 255)

print('Writing to disk ...')
## Resampling at the desired resolution
inplane_factor = INIT_RES / IMAGE_RES
aux = np.asarray([(inplane_factor - 1) / (2 * inplane_factor)] * 2 + [0])
cog_xy = -np.dot(TrotInv[:3, :3], MRI_AFFINE[:3, 3] - np.dot(MRI_AFFINE[:3, :3], aux.T))[1:]

if colored:
    vol, aff = build_regular_space_volume_color(data_H_eq, z_pos, cog_xy, IMAGE_RES, target_sp=zres)
    vol = (vol-150) * 255/105
    vol = np.clip(vol, 0, 255)
else:
    vol, aff = build_regular_space_volume(data_H_eq, z_pos, cog_xy, IMAGE_RES, target_sp=zres)

vol = vol.astype('uint8')
img = nib.Nifti1Image(vol, np.matmul(TrotInv, aff))
nib.save(img, join(dirname(filepath), filename + '.3D.' + str(zres) + '.nii.gz'))

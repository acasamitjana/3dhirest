# py
from os import makedirs
from os.path import join, exists, basename, dirname
import subprocess
import copy

# libraries imports
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import cv2
import torch
import torch.nn.functional as nnf

# project imports
from setup import *
from database import read_slice_info
from src.utils.io import create_results_dir

##################
###### Paths #####
##################

NiftyRegPath = NIFTY_REG_DIR
ALADINcmd = NiftyRegPath + 'reg-apps' + '/reg_aladin'
TRANSFORMcmd = NiftyRegPath + 'reg-apps/reg_transform'
REScmd = NiftyRegPath + 'reg-apps' + '/reg_resample'

INIT_BASE_DIR = join(DATA_DIR, 'downloads')
INIT_BASE_DIR_MRI = join(INIT_BASE_DIR, 'mri')
INIT_BASE_DIR_IHC = join(INIT_BASE_DIR, 'ihc')
INIT_BASE_DIR_NISSL = join(INIT_BASE_DIR, 'nissl')
INIT_BASE_DIR_LINEAL = join(INIT_BASE_DIR, 'linear')

MRI_ORIG = join(INIT_BASE_DIR_LINEAL, 'mri.orig.nii.gz')
MRI_LIN = join(INIT_BASE_DIR_LINEAL, 'mri.nii.gz')
HISTO_LIN = join(INIT_BASE_DIR_LINEAL, 'stack_nissl.lin.tree.nii.gz')


BASE_DIR = join(DATA_DIR, 'dataset')
BASE_DIR_MRI = join(BASE_DIR, 'mri')
BASE_DIR_IHC = join(BASE_DIR, 'ihc')
BASE_DIR_NISSL = join(BASE_DIR, 'nissl')

create_results_dir(BASE_DIR_MRI, subdirs=['images', 'masks'])
create_results_dir(BASE_DIR_IHC, subdirs=['images_resize', 'masks_resize', 'images', 'masks', 'affine'])
create_results_dir(BASE_DIR_NISSL, subdirs=['images_resize', 'masks_resize', 'images', 'masks', 'affine'])

#######################
###### Parameters #####
#######################
MRI_RES = 0.5
HISTO_RES = 0.032
HISTO_THICKNESS = 0.05
OUTPUT_RES = 0.25

BLOCKS = {'B1': (0, 166), 'B2':(166, 317), 'B3':(317 ,394), 'B4':(394, 472), 'B5':(472, 552), 'B6':(552, 641)}

file = 'slice_separation.csv'
slice_num_dict = read_slice_info(file, stain=['MRI', 'IHC', 'NISSL'])

proxy = nib.load(HISTO_LIN)
histo_shape_rigid = proxy.shape # Is that needed?

z_pos = np.load(join(INIT_BASE_DIR_LINEAL, 'z_pos.npy'))
#########################
###### Resample MRI #####
#########################
Trot = np.array([[0, 0, -1, 0], [-1, 0, 0, 0],  [0, -1, 0, 0], [0, 0, 0, 1]]) # brings MRI into the orientation of the stack
proxy = nib.load(MRI_LIN)
MRI_AFFINE = copy.copy(proxy.affine)
MRI_AFFINE = np.matmul(Trot, MRI_AFFINE)

proxy = nib.load(MRI_ORIG)
mri_vol = np.asarray(proxy.dataobj)

inplane_factor = [MRI_RES/OUTPUT_RES]*2
inplane_shape = tuple([int(np.round(a * b)) for a, b in zip(histo_shape_rigid[:2], inplane_factor)])
nslices = histo_shape_rigid[2] - 50 #25 emtpy slices A-P on each side

def resample_mri(mri_vol, inplane_shape, nslices, MRI_AFFINE, OUTPUT_RES, z_pos):

    mri_mask_bool = mri_vol > 0
    aux = mri_vol[mri_mask_bool]
    mini = np.quantile(aux, 0.001)
    maxi = np.quantile(aux, 0.999)

    mri_rearranged = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(mri_vol).float(), dim=0), dim=0)
    mri_mask_rearranged = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(mri_mask_bool).float(), dim=0), dim=0)

    cog_ij = [(ishape+1)/2 for ishape in inplane_shape]
    vectors = [torch.arange(0, s) for s in inplane_shape + (nslices,)]
    vectors[0] = vectors[0] - cog_ij[0]
    vectors[1] = vectors[1] - cog_ij[1]
    vectors[2] = vectors[2] + 0.0  # make float like the other
    grids = torch.stack(torch.meshgrid(vectors))

    grids_new_mri = torch.zeros(grids.shape)
    mri_aff_combined = torch.from_numpy(MRI_AFFINE).float()
    D = torch.inverse(mri_aff_combined)
    for d in range(3):
        grids_new_mri[d, :, :, :] = D[d, 0] * OUTPUT_RES * grids[0, :, :, :] \
                                    + D[d, 1] * OUTPUT_RES * grids[1, :, :, :] \
                                    + D[d, 2] * torch.unsqueeze(torch.unsqueeze(torch.from_numpy(z_pos).float(), 0), 0) \
                                    + D[d, 3]

    grids_new_mri = torch.unsqueeze(grids_new_mri, 0)
    grids_new_mri = grids_new_mri.permute(0, 2, 3, 4, 1)
    for i in range(3):
        grids_new_mri[:, :, :, :, i] = 2 * (grids_new_mri[:, :, :, :, i] / (mri_vol.shape[i] - 1) - 0.5)

    # Not sure why, but channels need to be reversed
    grids_new_mri = grids_new_mri[..., [2, 1, 0]]
    mri_resampled = nnf.grid_sample(mri_rearranged, grids_new_mri, align_corners=True, mode='bilinear', padding_mode='zeros')
    mri_resampled = torch.squeeze(mri_resampled)
    mri_resampled_numpy = mri_resampled.cpu().detach().numpy()

    mri_mask_resampled = nnf.grid_sample(mri_mask_rearranged, grids_new_mri, align_corners=True, mode='bilinear', padding_mode='zeros')
    mri_mask_resampled = torch.squeeze(mri_mask_resampled)
    mri_mask_resampled_numpy = mri_mask_resampled.cpu().detach().numpy()

    mri_resampled_numpy[mri_mask_resampled_numpy<0.5] = 0
    mri_resampled_numpy = np.clip(mri_mask_resampled_numpy * (mri_resampled_numpy - mini) / (maxi - mini), 0, 1)

    return mri_resampled_numpy, mri_mask_resampled_numpy


mri_resampled_numpy, mri_mask_resampled_numpy = \
    resample_mri(mri_vol, inplane_shape, nslices, MRI_AFFINE, OUTPUT_RES, z_pos)

############################
###### Generate slices #####
############################
stain_path_dict = {'MRI': [INIT_BASE_DIR_MRI, BASE_DIR_MRI],
                   'NISSL': [INIT_BASE_DIR_NISSL, BASE_DIR_NISSL],
                   'IHC': [INIT_BASE_DIR_IHC ,BASE_DIR_IHC]}
for stain, stain_dict in slice_num_dict.items():

    print('\n\n' + stain)
    print('   Slice number:', end=' ', flush=True)

    INIT_DATA_DIR = stain_path_dict[stain][0]
    DATA_DIR = stain_path_dict[stain][1]

    for slice_file, slice_features in stain_dict.items():
        slice_tree = int(slice_features['tree_pos'])
        slice_str = slice_file.split('.')[0]
        sid = int(slice_str.split('_')[1])

        print('#' + str(sid), end=' ', flush=True)
        if stain == 'MRI':
            mri_slice = np.squeeze(mri_resampled_numpy[..., slice_tree])[93:93+448,79:79+320]
            mri_mask_slice = np.squeeze(mri_mask_resampled_numpy[..., slice_tree])[93:93+448,79:79+320]

            mri_slice = (255 * mri_slice).astype(np.uint8)
            mri_mask_slice = (255 * mri_mask_slice).astype(np.uint8)

            img = Image.fromarray(mri_slice, mode='L')
            img.save(join(DATA_DIR, 'images', slice_file))

            img = Image.fromarray(mri_mask_slice, mode='L')
            img.save(join(DATA_DIR, 'masks', slice_file))

        else:
            histo_filepath = join(INIT_DATA_DIR, 'images', slice_file)
            histo_mask_filepath = join(INIT_DATA_DIR, 'masks', slice_file)

            H = cv2.imread(histo_filepath)
            M = cv2.imread(histo_mask_filepath, flags=0)
            if np.max(M) == 0 or M is None:
                continue
            M = (M / np.max(M)) > 0.5
            for it_c in range(3):
                htmp = H[..., it_c]
                htmp[~M] = 0
                H[..., it_c] = htmp

            H_gray = cv2.cvtColor(H, cv2.COLOR_RGB2GRAY)
            H_gray[~M] = 0
            H_gray[M] = 255 - H_gray[M]
            H_gray = gaussian_filter((H_gray/255).astype(np.double), sigma=5)
            H_gray[M] = (H_gray[M] - np.min(H_gray[M])) /(np.max(H_gray[M]) - np.min(H_gray[M]))

            H_gray = (255*H_gray).astype('uint8')
            M = (255*M).astype('uint8')

            # image_preprocess = []
            # import pdb
            # for it_image, image in enumerate([H, H_gray, M]):
            #     resized_shape = tuple([int(i * HISTO_RES / OUTPUT_RES) for i in image.shape[:2]])
            #     pdb.set_trace()
            #     image = resize(image, resized_shape, anti_aliasing=True)
            #
            #     image = np.double(image)
            #     image = image/255
            #     image_preprocess.append(image)

            resized_shape = tuple([int(i * HISTO_RES / OUTPUT_RES) for i in M.shape[:2]])
            H, H_gray, M = [ resize(image, resized_shape, anti_aliasing=True) for image in [H, H_gray, M]]

            H = (255 * H).astype('uint8')
            H = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)

            img = Image.fromarray((255*M).astype('uint8'), mode='L')
            img.save(join(DATA_DIR, 'masks_resize', slice_file))

            img = Image.fromarray((255*H_gray).astype('uint8'), mode='L')
            img.save(join(DATA_DIR, 'images_resize', slice_file))

            # Affine registration
            refFile = join(BASE_DIR_MRI, 'images', 'slice_' + "{:03d}".format(slice_tree+1) + '.png')
            refMaskFile = join(BASE_DIR_MRI, 'masks', 'slice_' + "{:03d}".format(slice_tree+1) + '.png')

            floGrayFile = join(DATA_DIR, 'images_resize', slice_file)
            floMaskFile = join(DATA_DIR, 'masks_resize', slice_file)

            outputGrayFile = join(DATA_DIR, 'images', slice_file)
            outputMaskFile = join(DATA_DIR, 'masks', slice_file)
            affineFile = join(DATA_DIR, 'affine', slice_str + '.aff')

            subprocess.call(
                [ALADINcmd, '-ref', refFile, '-flo', floGrayFile, '-aff', affineFile, '-res', outputGrayFile,
                 '-ln', '4', '-lp', '3', '-pad', '0', '-speeeeed'], stdout=subprocess.DEVNULL)

            subprocess.call(
                [REScmd, '-ref', refFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile,
                 '-inter', '0', '-voff'], stdout=subprocess.DEVNULL)

    directories = [join(DATA_DIR, 'masks'), join(DATA_DIR, 'images')]

    for d in directories:

        files = list(slice_num_dict[stain].keys())
        for f, slice_info in slice_num_dict[stain].items():

            im = cv2.imread(join(d, f))
            if im is None:
                continue

            if len(im.shape) > 2:
                im = im[..., 0]

            if 'vol' not in locals():
                vol = np.zeros((im.shape[0], im.shape[1], 641))

            vol[..., int(slice_info['tree_pos'])] = im

        vol = np.asarray(vol)
        img = nib.Nifti1Image(vol, np.eye(4))
        nib.save(img, join(DATA_DIR, stain + '_' + basename(d) + '.nii.gz'))

        for b, nk in BLOCKS.items():
            nku = nk[0]
            nkd = nk[1]
            pathb = join(dirname(DATA_DIR), b)
            if not exists(pathb):
                makedirs(pathb)
            datab = vol[..., nku:nkd]
            img = nib.Nifti1Image(datab, proxy.affine)
            nib.save(img, join(pathb, stain + '_' + basename(d) + '.nii.gz'))

        del vol


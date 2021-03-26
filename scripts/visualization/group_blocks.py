import os
from os.path import join, exists
import numpy as np
import nibabel as nib

from setup import DATA_DIR
from database.data_loader import DataLoaderBlock
from scripts import config_dev as configFile, config_data
from src.utils.algorithm_utils import integrate_RegNet, integrate_NR
from src.utils.image_utils import build_regular_space_volume

# Paths
input_z_pos = join(DATA_DIR, 'dataset', 'linear', 'z_pos.npy')
data_path_dict = {
    'RegNet': config_data.OBSERVATIONS_DIR_REGNET,
    'NR': config_data.OBSERVATIONS_DIR_NR,
    'ST3_L1_RegNet': join(config_data.ALGORITHM_DIR, 'ST3_RegNet', 'l1', 'NN4'),
    'ST3_L2_RegNet': join(config_data.ALGORITHM_DIR, 'ST3_RegNet', 'l2', 'NN4'),
    'ST3_L1_NR': join(config_data.ALGORITHM_DIR, 'ST3_NR', 'l1', 'NN4'),
    'ST3_L2_NR': join(config_data.ALGORITHM_DIR, 'ST3_NR', 'l2', 'NN4')
}

reg_algorithm_list = ['ST3_L1_RegNet']

# Parameters
Z_RES = 0.25

INIT_RES = config_data.INIT_RES
MRI_AFFINE = config_data.MRI_AFFINE
BLOCK_RES = configFile.BLOCK_RES
CROP_XY = configFile.CROP_XY
Trot = np.array([[0, 0, -1, -CROP_XY[0]], [-1, 0, 0, -CROP_XY[1]],  [0, -1, 0, 0], [0, 0, 0, 1]]) # brings MRI into the orientation of the stack
TrotInv = np.linalg.inv(Trot)

inplane_factor = INIT_RES / BLOCK_RES
aux = np.asarray([(inplane_factor - 1) / (2 * inplane_factor)]*2 + [0])
cog_xy = np.dot(TrotInv[:3,:3], MRI_AFFINE[:3, 3] - np.dot(MRI_AFFINE[:3, :3], aux.T))

z_pos = 1 * np.squeeze(np.load(input_z_pos))

BLOCKS = ['B1','B2','B3','B4', 'B5', 'B6']
stain_list = ['NISSL', 'IHC']
for reg_algorithm in reg_algorithm_list:
    path = data_path_dict[reg_algorithm]
    for stain in stain_list:
        print('\n' + stain)
        block_dict = {}
        flow_dict = {}
        flow_reverse_dict = {}
        nslices = [0]
        for block in BLOCKS:
            print('\n  -' + block)
            proxy = nib.load(join(path, block, stain + '.tree.nii.gz'))
            block_dict[block] = np.asarray(proxy.dataobj)
            nslices += [nslices[-1] + block_dict[block].shape[-1]]
            if block == BLOCKS[0]:
                affine = proxy.affine
                xy_shape = block_dict[block].shape[:2]

            # Flow
            if not exists(join(path, block, stain + '.flow.tree.nii.gz')):
                proxy_vel = nib.load(join(path, block, stain + '.field_x.tree.nii.gz'))
                vel = np.zeros((2,) + proxy_vel.shape)
                vel[0] = np.asarray(proxy_vel.dataobj)

                proxy_vel = nib.load(join(path, block, stain + '.field_y.tree.nii.gz'))
                vel[1] = np.asarray(proxy_vel.dataobj)

                data_loader = DataLoaderBlock(configFile.CONFIG_DICT['MRI'])
                block_list = data_loader.subject_list
                block_shape = data_loader.image_shape
                if reg_algorithm == 'RegNet':
                    flow = integrate_RegNet(vel, block_shape, configFile.CONFIG_DICT['BASE'])

                else:
                    flow = integrate_NR(vel, block_shape)

                flow_dict[block] = flow
                img = nib.Nifti1Image(flow, affine)
                nib.save(img, join(path, block, stain + '.flow.tree.nii.gz'))

            else:
                proxy = nib.load(join(path, block, stain + '.flow.tree.nii.gz'))
                flow_dict[block] = np.asarray(proxy.dataobj)


            # Flow reverse
            if not exists(join(path, block, stain + '.flow.reverse.tree.nii.gz')):
                proxy_vel = nib.load(join(path, block, stain + '.field_x.tree.nii.gz'))
                vel = np.zeros((2,) + proxy_vel.shape)
                vel[0] = np.asarray(proxy_vel.dataobj)

                proxy_vel = nib.load(join(path, block, stain + '.field_y.tree.nii.gz'))
                vel[1] = np.asarray(proxy_vel.dataobj)

                data_loader = DataLoaderBlock(configFile.CONFIG_DICT['MRI'])
                block_list = data_loader.subject_list
                block_shape = data_loader.image_shape
                if reg_algorithm == 'RegNet':
                    flow = integrate_RegNet(-vel, block_shape, configFile.CONFIG_DICT['BASE'])

                else:
                    flow = integrate_NR(-vel, block_shape)

                flow_reverse_dict[block] = flow
                img = nib.Nifti1Image(flow, affine)
                nib.save(img, join(path, block, stain + '.flow.reverse.tree.nii.gz'))

            else:
                proxy = nib.load(join(path, block, stain + '.flow.reverse.tree.nii.gz'))
                flow_reverse_dict[block] = np.asarray(proxy.dataobj)


        vol = np.zeros(xy_shape + (nslices[-1],))
        flow = np.zeros((2,) + xy_shape + (nslices[-1],))
        flow_reverse = np.zeros((2,) + xy_shape + (nslices[-1],))
        for it_block, block in enumerate(BLOCKS):
            vol[..., nslices[it_block]:nslices[it_block+1]] = block_dict[block]
            flow[..., nslices[it_block]:nslices[it_block+1]] = flow_dict[block]
            flow_reverse[..., nslices[it_block]:nslices[it_block+1]] = flow_reverse_dict[block]

        img = nib.Nifti1Image(vol, affine)
        nib.save(img, join(path, stain + '.tree.nii.gz'))

        vol, aff = build_regular_space_volume(vol, z_pos, cog_xy, BLOCK_RES, target_sp=Z_RES, max_z_error=0.1)

        img = nib.Nifti1Image(vol, np.matmul(TrotInv, aff))
        nib.save(img,join(path, stain + '.resampled_' + str(Z_RES) + '.nii.gz'))

        img = nib.Nifti1Image(flow, affine)
        nib.save(img,join(path, stain + '.flow.tree.nii.gz'))

        img = nib.Nifti1Image(flow_reverse, affine)
        nib.save(img,join(path, stain + '.flow.reverse.tree.nii.gz'))
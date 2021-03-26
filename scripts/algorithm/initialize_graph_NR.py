# imports
from os.path import join, exists
from os import makedirs
import time
from argparse import ArgumentParser

# third party imports
import numpy as np
import nibabel as nib

# project imports
from database.data_loader import DataLoaderBlock
from database.databaseConfig import Allen_MRI
from src import datasets
from scripts import config_dev as configFile, config_data
from src.utils.algorithm_utils import initialize_graph_NR

SCONTROL = [-8, -8]
HISTO_AFFINE = configFile.HISTO_AFFINE
results_dir = config_data.OBSERVATIONS_DIR_NR
tempdir = join(results_dir, 'tmp')
if not exists(tempdir):
    makedirs(tempdir)

####################################
############ PARAMETERS ############
####################################

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--nn', default=4, type=int)
arguments = arg_parser.parse_args()
N_NEIGHBOURS = arguments.nn

#######################################
########### Tree parameters ###########
#######################################

print('Loading dataset ...\n')
data_loader_MRI = DataLoaderBlock(configFile.CONFIG_MRI)
prefix_list = ['MRI_NISSL', 'MRI_IHC', 'IHC_NISSL']
num_neighbours_list = [0]*3
for it_n in range(N_NEIGHBOURS):
    prefix_list.extend(['MRI', 'NISSL', 'IHC'])
    num_neighbours_list.extend([it_n+1]*3)

########################################
########### Run registration ###########
########################################
for it_network in range(3 * (1 + N_NEIGHBOURS)):
    stain = prefix_list[it_network]
    num_neighbours = num_neighbours_list[it_network]
    parameter_dict = configFile.CONFIG_DICT[stain]

    data_loader = DataLoaderBlock(parameter_dict)
    subject_list = data_loader.subject_list
    rid_list = data_loader.rid_list
    nsubjects = len(data_loader)
    image_shape = data_loader.image_shape

    print('Registering: ' + stain + ' ...')

    num_tree_pos_prev = 0
    for it_sbj, sbj_general in enumerate(subject_list):
        print('    - Block: ' + sbj_general.id)

        sbj = data_loader.subject_dict[sbj_general.id]
        data_loader.subject_list = sbj.slice_list
        data_loader.rid_list = [s.id for s in sbj.slice_list]
        nslices = len(sbj.slice_list)

        results_dir_sbj = join(results_dir, sbj.id)
        if not exists(results_dir_sbj):
            makedirs(results_dir_sbj)

        if it_network < 3:
            # intermodal
            dataset = datasets.InterModalRegistrationDataset(
                data_loader,
                rotation_params=parameter_dict['ROTATION'],
                nonlinear_params=parameter_dict['NONLINEAR'],
                tf_params=parameter_dict['TRANSFORM'],
                da_params=parameter_dict['DATA_AUGMENTATION'],
                norm_params=parameter_dict['NORMALIZATION'],
                mask_dilation=np.ones((15, 15)),
                to_tensor=False
            )

        else:
            # intramodal
            dataset = datasets.IntraModalRegistrationDataset(
                data_loader,
                rotation_params=parameter_dict['ROTATION'],
                nonlinear_params=parameter_dict['NONLINEAR'],
                tf_params=parameter_dict['TRANSFORM'],
                da_params=parameter_dict['DATA_AUGMENTATION'],
                norm_params=parameter_dict['NORMALIZATION'],
                mask_dilation=np.ones((15, 15)),
                neighbor_distance=-num_neighbours,
                fix_neighbors=True,
                to_tensor=False
            )


        #######################################
        ############ Registration #############
        #######################################
        filename = stain + '.' + str(num_neighbours) + 'N'
        if exists(join(results_dir_sbj, filename + '.flow.nii.gz')):
            num_tree_pos_prev = data_loader_MRI.subject_dict[sbj_general.id].num_tree_pos
            continue

        t_init = time.time()

        output_results = initialize_graph_NR(dataset, image_shape, SCONTROL, tempdir=tempdir)
        registered_image, registered_mask, velocity_field, displacement_field = output_results

        print('NR elapsed time: ' + str(np.round(time.time() - t_init, 2)))


        # Save output forward tree
        num_tree_pos = data_loader_MRI.subject_dict[sbj.id].num_tree_pos - num_tree_pos_prev
        registered_image_tree = np.zeros(registered_image.shape[:2] + (num_tree_pos,))
        registered_mask_tree = np.zeros(registered_image.shape[:2] + (num_tree_pos,))
        velocity_field_x_tree = np.zeros(velocity_field.shape[1:3] + (num_tree_pos,))
        velocity_field_y_tree = np.zeros(velocity_field.shape[1:3] + (num_tree_pos,))

        for it_sl, sl in enumerate(data_loader.subject_list):
            if it_sl >= nslices - num_neighbours:
                break
            registered_image_tree[..., sl.tree_pos - num_tree_pos_prev] = registered_image[..., it_sl]
            registered_mask_tree[..., sl.tree_pos - num_tree_pos_prev] = registered_mask[..., it_sl]
            velocity_field_x_tree[..., sl.tree_pos - num_tree_pos_prev] = velocity_field[0, ..., it_sl]
            velocity_field_y_tree[..., sl.tree_pos - num_tree_pos_prev] = velocity_field[1, ..., it_sl]

        img = nib.Nifti1Image(registered_image_tree, HISTO_AFFINE)
        nib.save(img, join(results_dir_sbj, filename + '.tree.nii.gz'))

        img = nib.Nifti1Image(registered_mask_tree, HISTO_AFFINE)
        nib.save(img, join(results_dir_sbj, filename + '.mask.tree.nii.gz'))

        img = nib.Nifti1Image(velocity_field_x_tree, HISTO_AFFINE)
        nib.save(img, join(results_dir_sbj, filename + '.field_x.tree.nii.gz'))

        img = nib.Nifti1Image(velocity_field_y_tree, HISTO_AFFINE)
        nib.save(img, join(results_dir_sbj, filename + '.field_y.tree.nii.gz'))

        num_tree_pos_prev = data_loader_MRI.subject_dict[sbj_general.id].num_tree_pos

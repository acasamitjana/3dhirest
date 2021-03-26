from os.path import join, exists
from os import makedirs
import time
from argparse import ArgumentParser

# project imports
from database.data_loader import DataLoaderBlock
from database import read_slice_info
from src import datasets
from src.utils import algorithm_utils
from src.utils.image_utils import deform2D
from scripts import config_dev as configFile, config_data
from src.algorithm import *

file = 'slice_separation_extended.csv'
slice_num_dict = read_slice_info(file, stain=['MRI', 'IHC', 'NISSL'])
affine = configFile.HISTO_AFFINE

parameter_dict_MRI = configFile.CONFIG_DICT['MRI']
parameter_dict_NISSL = configFile.CONFIG_DICT['NISSL']
parameter_dict_IHC = configFile.CONFIG_DICT['IHC']

if __name__ == '__main__':

    # Parameters
    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--c1', type=str, default='IHC', choices=['IHC', 'NISSL'], help='Reference volume modality')
    arg_parser.add_argument('--c2', type=str, default='NISSL', choices=['IHC', 'NISSL', ''], help='Reference volume modality')
    arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
    arg_parser.add_argument('--nn', type=int, default=2, help='Number of neighbours')
    arg_parser.add_argument('--mdil', type=int, default=7, help='Mask dilation factor')

    arguments = arg_parser.parse_args()
    ref = 'MRI'
    c1 = arguments.c1
    c2 = arguments.c2
    nneighbours = arguments.nn
    cost = arguments.cost
    mdil = arguments.mdil

    if c2 == '':
        N_CONTRASTS = 2
    else:
        N_CONTRASTS = 3


    observations_dir = config_data.OBSERVATIONS_DIR_NR
    algorithm_dir = config_data.ALGORITHM_DIR
    results_dir = join(algorithm_dir, 'NiftyReg', cost, 'NN' + str(nneighbours))
    if not exists(results_dir):
        makedirs(results_dir)

    data_loader = DataLoaderBlock(parameter_dict_MRI)
    block_list = data_loader.subject_list
    block_shape = data_loader.image_shape

    dataset = datasets.IntraModalRegistrationDataset(
        data_loader,
        rotation_params=parameter_dict_MRI['ROTATION'],
        nonlinear_params=parameter_dict_MRI['NONLINEAR'],
        tf_params=parameter_dict_MRI['TRANSFORM'],
    )
    image_shape = dataset.image_shape
    cp_shape = tuple([int(i / parameter_dict_MRI['UPSAMPLE_LEVELS']) for i in image_shape])

    num_tree_pos_prev = 0
    print('[START] Processing')
    for it_sbj, sbj in enumerate(block_list):
        print('   - Subject: ' + str(sbj.id))
        input_dir = join(observations_dir, sbj.id)
        subject_dir = join(parameter_dict_MRI['DB_CONFIG']['BASE_DIR'], sbj.id)

        if not exists(join(input_dir, c1 + '.' + nneighbours + 'N.field_x.tree.nii.gz')):
            print('[WARNING] No observations found for subject ' + sbj.id + ', contrast ' + c1 +  ' and NiftyReg ')
            continue

        nslices = len(sbj.slice_list)
        results_dir_sbj = join(results_dir, sbj.id)
        if not exists(join(results_dir_sbj)):
            makedirs(results_dir_sbj)
        elif exists(join(results_dir_sbj, c1 + '.nii.gz')):
            print('[DONE] Subject ' + sbj.id + ' has already been processed')
            num_tree_pos_prev += nslices
            continue


        ####################################################################################################
        ####################################################################################################

        print('[Init Graph] Reading SVFs ...')
        t_init = time.time()
        if N_CONTRASTS == 2:
            graph_structure = init_st2(subject_dir, input_dir, cp_shape, nslices,
                                       nneighbours=nneighbours, se=np.ones((mdil, mdil)))

            R, M, W, d_inter, d_Ref, d_C1, NK = graph_structure
            print('[Init Graph] Total Elapsed time: ' + str(time.time() - t_init))

            print('[ALGORITHM] Running the algorithm ...')
            if cost == 'L2':
                Tres = st2_L2(R, M, W, d_inter, d_Ref, d_C1, nslices, niter=5)

            else:
                Tres = st2_L1(R, M, W, nslices)

            T_C1 = Tres[..., :nslices]
            T_Ref = Tres[..., nslices:]

            img = nib.Nifti1Image(T_C1[0], affine)
            nib.save(img, join(results_dir_sbj, c1 + '.field_x.tree.nii.gz'))
            img = nib.Nifti1Image(T_C1[1], affine)
            nib.save(img, join(results_dir_sbj, c1 + '.field_y.tree.nii.gz'))

            img = nib.Nifti1Image(T_Ref[0], affine)
            nib.save(img, join(results_dir_sbj, ref + '.field_x.tree.nii.gz'))
            img = nib.Nifti1Image(T_Ref[1], affine)
            nib.save(img, join(results_dir_sbj, ref + '.field_y.tree.nii.gz'))


        elif N_CONTRASTS == 3:
            graph_structure = init_st3(subject_dir, input_dir, cp_shape, nslices,
                                       nneighbours=nneighbours, se=np.ones((mdil, mdil)))

            R, M, W, d_inter, d_Ref, d_C1, d_C2, NK = graph_structure
            print('[Init Graph] Total Elapsed time: ' + str(time.time() - t_init))

            print('[ALGORITHM] Running the algorithm ...')
            if cost == 'L2':
                Tres = st3_L2(R, M, W, d_inter, d_Ref, d_C1, d_C2, nslices, niter=5)

            else:
                Tres = st3_L1(R, M, W, nslices)

            T_C1 = Tres[..., :nslices]
            T_C2 = Tres[..., nslices:2 * nslices]
            T_Ref = Tres[..., 2 * nslices:]

            img = nib.Nifti1Image(T_C1[0], affine)
            nib.save(img, join(results_dir_sbj, c1 + '.field_x.tree.nii.gz'))
            img = nib.Nifti1Image(T_C1[1], affine)
            nib.save(img, join(results_dir_sbj, c1 + '.field_y.tree.nii.gz'))

            img = nib.Nifti1Image(T_C2[0], affine)
            nib.save(img, join(results_dir_sbj, c2 + '.field_x.tree.nii.gz'))
            img = nib.Nifti1Image(T_C2[1], affine)
            nib.save(img, join(results_dir_sbj, c2 + '.field_y.tree.nii.gz'))

            img = nib.Nifti1Image(T_Ref[0], affine)
            nib.save(img, join(results_dir_sbj, ref + '.field_x.tree.nii.gz'))
            img = nib.Nifti1Image(T_Ref[1], affine)
            nib.save(img, join(results_dir_sbj, ref + '.field_y.tree.nii.gz'))

        else:
            raise ValueError("[ERROR] The number of contrasts specified is not valid")

        ####################################################################################################
        ####################################################################################################
        print('[INTEGRATION] Computing deformation field ... ')
        t_init = time.time()
        if not exists(join(results_dir_sbj, c1 + '.flow.nii.gz')):
            flow_c1 = algorithm_utils.integrate_NR(T_C1, block_shape)

            img = nib.Nifti1Image(flow_c1, affine)
            nib.save(img, join(results_dir_sbj, c1 + '.flow.nii.gz'))

        if N_CONTRASTS==3 and not exists(join(results_dir_sbj, c2 + '.flow.nii.gz')):
            flow_c2 = algorithm_utils.integrate_NR(T_C2, block_shape)

            img = nib.Nifti1Image(flow_c2, affine)
            nib.save(img, join(results_dir_sbj, c2 + '.flow.nii.gz'))

        print('[INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))

        ####################################################################################################
        ####################################################################################################

        print('[DEFORM] Deforming images ... ')

        # IHC
        if c1 == 'IHC' or c2 == 'IHC':
            data_loader_IHC = DataLoaderBlock(parameter_dict_IHC)
            block_list_IHC = data_loader_IHC.subject_list

            proxy = nib.load(join(results_dir_sbj, 'IHC.flow.nii.gz'))
            flow = np.asarray(proxy.dataobj)

            image_deformed = np.zeros(block_shape + (nslices,))
            mask_deformed = np.zeros(block_shape + (nslices,))

            for sl in block_list_IHC[it_sbj].slice_list:
                it_sl = sl.tree_pos - num_tree_pos_prev
                print('         Slice: ' + str(it_sl) + '/' + str(nslices))

                image = sl.load_ihc()
                mask = sl.load_ihc_mask()

                image_deformed[..., it_sl] = deform2D(image, flow[..., it_sl])
                mask_deformed[..., it_sl] = deform2D(mask, flow[..., it_sl], mode='nearest')

                del image
                del mask

            img = nib.Nifti1Image(image_deformed, affine)
            nib.save(img, join(results_dir_sbj, 'IHC.nii.gz'))

            img = nib.Nifti1Image(mask_deformed, affine)
            nib.save(img, join(results_dir_sbj, 'IHC.mask.nii.gz'))


        if c1 == 'NISSL' or c2 == 'NISSL':
            # NISSL
            data_loader_NISSL = DataLoaderBlock(parameter_dict_NISSL)
            block_list_NISSL = data_loader_NISSL.subject_list

            proxy = nib.load(join(results_dir_sbj, 'NISSL.flow.nii.gz'))
            flow = np.asarray(proxy.dataobj)

            image_deformed = np.zeros(block_shape + (nslices,))
            mask_deformed = np.zeros(block_shape + (nslices,))

            for sl in block_list_NISSL[it_sbj].slice_list:
                it_sl = sl.tree_pos - num_tree_pos_prev
                print('         Slice: ' + str(it_sl) + '/' + str(nslices))

                image = sl.load_nissl()
                mask = sl.load_nissl_mask()

                # NISSL
                image_deformed[..., it_sl] = deform2D(image, flow[..., it_sl])
                mask_deformed[..., it_sl] = deform2D(mask, flow[..., it_sl], mode='nearest')


            img = nib.Nifti1Image(image_deformed, affine)
            nib.save(img, join(results_dir_sbj, 'NISSL.nii.gz'))

            img = nib.Nifti1Image(mask_deformed, affine)
            nib.save(img, join(results_dir_sbj, 'NISSL.mask.nii.gz'))


        num_tree_pos_prev += nslices

        print('[DEFORM] Total Elapsed time: ' + str(time.time() - t_init))


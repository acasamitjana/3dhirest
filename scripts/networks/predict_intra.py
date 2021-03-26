# imports
from os.path import join, exists
from os import makedirs

# third party imports
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser

# project imports
from src import  models, datasets
from database.data_loader import DataLoader
from src.utils import tensor_utils
from src.utils.io import create_results_dir, worker_init_fn
from src.utils.visualization import slices, plot_results
from scripts import config_dev as configFile
from src.test import predict

if __name__ == '__main__':

    ####################################
    ############ PARAMETERS ############
    ####################################
    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--epoch_number', default='FI', help='Load model from the epoch specified')
    arg_parser.add_argument('--modality', type=str, choices=['MRI', 'IHC', 'NISSL'], help='Modality(s) used in the registration')
    arg_parser.add_argument('--model', default='bidir', choices=['standard', 'bidir'])
    arg_parser.add_argument('--nsubjects', default=10, help='Number of subjects to predict')

    arguments = arg_parser.parse_args()
    epoch_weights = str(arguments.epoch_number)
    config_file_str = arguments.modality
    model_type = arguments.model
    number_of_subjects = arguments.nsubjects

    parameter_dict = configFile.CONFIG_DICT[config_file_str]
    parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + model_type

    use_gpu = torch.cuda.is_available() and parameter_dict['USE_GPU']
    device = torch.device("cuda:0" if use_gpu else "cpu")
    kwargs_generator = {'num_workers': 1, 'pin_memory': use_gpu, 'worker_init_fn': worker_init_fn}

    create_results_dir(parameter_dict['RESULTS_DIR'])


    ###################################
    ########### DATA LOADER ###########
    ###################################
    print('Loading dataset ...\n')
    data_loader = DataLoader(parameter_dict)
    sbj = data_loader.subject_list[0]
    nslices = len(sbj.slice_list)
    idx = np.random.choice(nslices)
    data_loader.subject_list = sbj.slice_list[idx:idx+number_of_subjects]
    data_loader.rid_list = [s.id for s in sbj.slice_list]

    dataset = datasets.IntraModalRegistrationDataset(
        data_loader,
        rotation_params=parameter_dict['ROTATION'],
        nonlinear_params=parameter_dict['NONLINEAR'],
        tf_params=parameter_dict['TRANSFORM'],
        da_params=parameter_dict['DATA_AUGMENTATION'],
        norm_params=parameter_dict['NORMALIZATION'],
        mask_dilation=np.ones((15, 15)),
        neighbor_distance=-parameter_dict['NEIGHBOR_DISTANCE'],
        fix_neighbors=True
    )


    generator_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameter_dict['BATCH_SIZE'],
        shuffle=False,
        **kwargs_generator
    )

    #################################
    ############# MDOEL #############
    #################################
    image_shape = dataset.image_shape
    input_channels = data_loader.n_channels * 2
    nonlinear_field_size = [9, 9]

    # Registration Network
    int_steps = 7
    registration = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=image_shape,
        int_steps=7,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        rescaling=True,
        gaussian_filter_flag=True,
    )
    registration = registration.to(device)

    da_model = tensor_utils.TensorDeformation(image_shape, nonlinear_field_size, device)

    epoch_results_dir = 'model_checkpoint.' + epoch_weights
    weightsfile = 'model_checkpoint.' + epoch_weights + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
    registration.load_state_dict(checkpoint['state_dict'])
    registration.eval()

    results_dir = join(parameter_dict['RESULTS_DIR'], 'results', epoch_results_dir)
    for bid in data_loader.subject_dict.keys():
        if not exists(join(results_dir, bid)):
            makedirs(join(results_dir, bid))

    results_file = join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv')
    if not exists(results_file):
        plot_results(results_file, ['loss_registration', 'loss_registration_smoothness', 'loss_magnitude', 'loss'])

    ref_image, flo_image, reg_image, flow_image, rid_list = predict(generator_test, registration, image_shape, device, da_model=da_model)

    for it_image, rid in enumerate(rid_list):
        print(str(it_image) + '/' + str(len(dataset)) + '. Slice: ' + str(rid))

        rid1, rid2 = rid.split('_to_')
        sid = rid1.split('.')[1] + '_' + rid2.split('.')[1]

        ref = ref_image[it_image]
        flo = flo_image[it_image]
        reg = reg_image[it_image]
        flow = flow_image[it_image]
        slices_2d = [flow[0], flow[1]]
        titles = ['FLOW_X image', 'FLOW_Y image']
        slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True, show=False)
        plt.savefig(join(results_dir, 'flow_' + sid + '.png'))
        plt.close()

        img_moving = Image.fromarray((255 * flo).astype(np.uint8), mode='L')
        img_fixed = Image.fromarray((255 * ref).astype(np.uint8), mode='L')
        img_registered = Image.fromarray((255 * reg).astype(np.uint8), mode='L')

        frames = [img_fixed, img_moving]
        frames[0].save(join(results_dir, 'initial_' + sid + '.gif'),
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=1000, loop=0)

        frames = [img_fixed, img_registered]
        frames[0].save(join(results_dir, 'fi_' + sid + '.gif'),
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=1000, loop=0)

        frames = [img_moving, img_registered]
        frames[0].save(join(results_dir, 'diff_' + sid + '.gif'),
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=1000, loop=0)

        print('Predicting done.')

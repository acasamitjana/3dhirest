#py
from argparse import ArgumentParser
from os.path import join
import time
from datetime import date, datetime

#external imports
import torch
import numpy as np

#project imports
from src import losses, models, datasets
from src.utils.io import ResultsWriter, create_results_dir, ExperimentWriter, worker_init_fn
from src.callbacks import History, ModelCheckpoint, PrinterCallback, ToCSVCallback, LRDecay
from src.utils import tensor_utils
from src.training import train, train_bidirectional
from src.utils.visualization import plot_results
from scripts import config_dev as configFile
from database.data_loader import DataLoader


####################################
############ PARAMETERS ############
####################################
date_start = date.today().strftime("%d/%m/%Y")
time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
intra_modal = ['MRI', 'IHC', 'NISSL']

if __name__ == '__main__':

    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--modality', type=str, choices=intra_modal, help='Modality(s) used in the registration')
    arg_parser.add_argument('--model', default='standard', choices=['standard' 'bidir'])
    arg_parser.add_argument('--mask', action='store_true')

    arguments = arg_parser.parse_args()
    config_file_str = arguments.modality
    mask_flag = arguments.mask
    model_type = arguments.model
    parameter_dict = configFile.CONFIG_DICT[config_file_str]

    use_gpu = torch.cuda.is_available() and parameter_dict['USE_GPU']
    device = torch.device("cuda:0" if use_gpu else "cpu")

    kwargs_training = {'mask_flag': mask_flag}
    kwargs_generator = {'num_workers': 1, 'pin_memory': use_gpu, 'worker_init_fn': worker_init_fn}

    parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + model_type
    create_results_dir(parameter_dict['RESULTS_DIR'])

    attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False
    resultsWriter = ResultsWriter(join(parameter_dict['RESULTS_DIR'], 'experiment_parameters.txt'), attach=attach)
    experimentWriter = ExperimentWriter(join(parameter_dict['RESULTS_DIR'], 'experiment.txt'), attach=attach)

    resultsWriter.write('Experiment parameters\n')
    for key, value in parameter_dict.items():
        resultsWriter.write(key + ': ' + str(value))
        resultsWriter.write('\n')
    resultsWriter.write('\n')

    ###################################
    ########### DATA LOADER ###########
    ###################################

    experimentWriter.write('Loading dataset ...\n')

    data_loader = DataLoader(parameter_dict)
    sbj = data_loader.subject_list[0]
    data_loader.subject_list = sbj.slice_list
    data_loader.rid_list = [s.id for s in sbj.slice_list]
    nslices = len(sbj.slice_list)

    dataset = datasets.IntraModalRegistrationDataset(
        data_loader,
        rotation_params=parameter_dict['ROTATION'],
        nonlinear_params=parameter_dict['NONLINEAR'],
        tf_params=parameter_dict['TRANSFORM'],
        da_params=parameter_dict['DATA_AUGMENTATION'],
        norm_params=parameter_dict['NORMALIZATION'],
        mask_dilation=np.ones((15, 15)),
        neighbor_distance=-parameter_dict['NEIGHBOR_DISTANCE'],
    )

    generator_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameter_dict['BATCH_SIZE'],
        shuffle=True,
        **kwargs_generator
    )

    #################################
    ############# MODEL #############
    #################################
    experimentWriter.write('Loading model ...\n')

    image_shape = dataset.image_shape
    nonlinear_field_size = [9, 9]
    da_model = tensor_utils.TensorDeformation(image_shape, nonlinear_field_size, device)

    int_steps = 7
    registration = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=image_shape,
        int_steps=int_steps,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        rescaling=True,
        gaussian_filter_flag=True,
    )

    registration = registration.to(device)
    optimizer = torch.optim.Adam(registration.parameters(), lr=parameter_dict['LEARNING_RATE'])

    if parameter_dict['STARTING_EPOCH'] > 0:
        weightsfile = 'model_checkpoint.' + str(parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
        checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
        optimizer.load_state_dict(checkpoint['optimizer'])
        registration.load_state_dict(checkpoint['state_dict'])

    # Losses
    reg_loss = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION']['name']]
    reg_loss = reg_loss(name='registration', device=device, **parameter_dict['LOSS_REGISTRATION']['params'])

    reg_smooth_loss = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['name']]
    reg_smooth_loss = reg_smooth_loss(name='registration_smoothness', loss_mult=parameter_dict['UPSAMPLE_LEVELS'],
                                      **parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['params'])

    loss_function_dict = {
        'registration': reg_loss,
        'registration_smoothness': reg_smooth_loss,
    }
    loss_weights_dict = {
        'registration': parameter_dict['LOSS_REGISTRATION']['lambda'],
        'registration_smoothness': parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'],
    }

    experimentWriter.write('Model ...\n')
    for name, param in registration.named_parameters():
        if param.requires_grad:
            experimentWriter.write(name + '. Shape:' + str(torch.tensor(param.data.size()).numpy()))
            experimentWriter.write('\n')

    ####################################
    ############# TRAINING #############
    ####################################
    experimentWriter.write('Training ...\n')
    experimentWriter.write('Number of images = ' + str(len(data_loader)))

    # Callbacks
    checkpoint_dir = join(parameter_dict['RESULTS_DIR'], 'checkpoints')
    results_file = join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv')
    log_keys = ['loss_' + lossname for lossname in loss_function_dict.keys()] + ['loss', 'time_duration (s)']

    logger = History(log_keys)
    training_printer = PrinterCallback()
    lrdecay = LRDecay(optimizer, n_iter_start=0, n_iter_finish=parameter_dict['N_EPOCHS'])
    model_checkpoint = ModelCheckpoint(checkpoint_dir, parameter_dict['SAVE_MODEL_FREQUENCY'])
    training_tocsv = ToCSVCallback(filepath=results_file, keys=log_keys)

    callback_list = [logger, model_checkpoint, training_printer, training_tocsv, lrdecay]

    for cb in callback_list:
        cb.on_train_init(registration, starting_epoch=parameter_dict['STARTING_EPOCH'])

    for epoch in range(parameter_dict['STARTING_EPOCH'], parameter_dict['N_EPOCHS']):
        epoch_start_time = time.time()

        logs_dict = {}
        for cb in callback_list:
            cb.on_epoch_init(registration, epoch)

        registration.train()
        if model_type == 'standard':
            train(registration, optimizer, device, generator_train, epoch, loss_function_dict,
                  loss_weights_dict, callback_list, da_model, **kwargs_training)

        elif model_type == 'bidir':
            train_bidirectional(registration, optimizer, device, generator_train, epoch, loss_function_dict,
                                loss_weights_dict, callback_list, da_model, **kwargs_training)

        else:
            raise ValueError("Please, specify a valid model_type")

        epoch_end_time = time.time()
        logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

        for cb in callback_list:
            cb.on_epoch_fi(logs_dict, registration, epoch, optimizer=optimizer)

    for cb in callback_list:
        cb.on_train_fi(registration)

    plot_results(results_file, keys=log_keys)

    print('Done.')

from os.path import join
import subprocess

import nibabel as nib
import torch
import numpy as np
from skimage.transform import resize
from PIL import Image

from setup import NIFTY_REG_DIR
from src import models
from src.utils import image_transform as tf
from src.utils.image_transform import bilinear_interpolate
from scripts import configFile

F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'

def initialize_graph_RegNet(model, generator_data, image_shape, device):

    num_elements = len(generator_data.dataset)
    num_batches = len(generator_data)
    batch_size = generator_data.batch_size

    downsample_factor = model.resize.factor
    vel_shape = tuple([int(i*d) for i,d in zip(image_shape,downsample_factor)])

    with torch.no_grad():

        registered_image = np.zeros((num_elements,) + image_shape)
        registered_mask = np.zeros((num_elements,) + image_shape)
        velocity_field = np.zeros((num_elements, 2) + vel_shape)
        deformation_field = np.zeros((num_elements, 2) + image_shape)

        for it_batch, data_list in enumerate(generator_data):

            start = it_batch * batch_size
            end = start + batch_size
            if it_batch == num_batches - 1:
                end = num_elements


            if torch.sum(data_list[1]) == 0 or torch.sum(data_list[3]) == 0:
                continue

            ref_image = data_list[0].to(device)
            flo_image = data_list[1].to(device)
            flo_mask = data_list[3].to(device)

            r, f, v = model(flo_image, ref_image)
            r_mask = model.predict(flo_mask, f, diffeomorphic=False, mode='nearest')

            registered_image[start:end] = np.squeeze(r.cpu().detach().numpy())
            registered_mask[start:end] = np.squeeze(r_mask.cpu().detach().numpy())
            velocity_field[start:end] = v.cpu().detach().numpy()

            deformation_field[start:end, 0] = f[:, 0].cpu().detach().numpy()
            deformation_field[start:end, 1] = f[:, 1].cpu().detach().numpy()
            deformation_field[start:end] = deformation_field[start:end]

    velocity_field[np.isnan(velocity_field)] = 0
    deformation_field[np.isnan(deformation_field)] = 0

    return np.transpose(registered_image, [1, 2, 0]), np.transpose(registered_mask, [1, 2, 0]), \
           np.transpose(velocity_field, [1, 2, 3, 0]), np.transpose(deformation_field, [1, 2, 3, 0]),\

def initialize_graph_NR(dataset, image_shape, scontrol, tempdir='/tmp'):

    # Filenames
    refFile = join(tempdir, 'refFile.png')
    floFile = join(tempdir, 'floFile.png')
    refMaskFile = join(tempdir, 'refMaskFile.png')
    floMaskFile = join(tempdir, 'floMaskFile.png')

    outputFile = join(tempdir, 'outputFile.png')
    outputMaskFile = join(tempdir, 'outputMaskFile.png')
    nonlinearField = join(tempdir, 'nonlinearField.nii.gz')
    dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')

    # Containers
    num_elements = len(dataset)
    registered_image = np.zeros((num_elements,) + image_shape)
    registered_mask = np.zeros((num_elements,) + image_shape)
    velocity_field = np.zeros((num_elements,2) + image_shape)
    displacement_field = np.zeros((num_elements,2) + image_shape)

    print('      Processing (N=' + str(num_elements) + '): ', end=' ', flush=True)
    nstep = 7
    # NiftyReg for all slices
    for it_batch in range(num_elements):
        print(str(it_batch), end=' ', flush=True)

        data_list = dataset[it_batch]

        x_ref = data_list[0]
        x_flo = data_list[1]
        m_ref = data_list[2]
        m_flo = data_list[3]
        if np.sum(m_ref) > 0 and np.sum(m_flo) > 0:

            #Save images
            img = Image.fromarray((255 * x_ref).astype(np.uint8), mode='L')
            img.save(refFile)
            img = Image.fromarray((255 * x_flo).astype(np.uint8), mode='L')
            img.save(floFile)
            img = Image.fromarray((255 * m_ref).astype(np.uint8), mode='L')
            img.save(refMaskFile)
            img = Image.fromarray((255 * m_flo).astype(np.uint8), mode='L')
            img.save(floMaskFile)

            # System calls
            subprocess.call([F3Dcmd, '-ref' , refFile , '-flo' , floFile , '-res', outputFile , '-cpp' , dummyFileNifti , '-sx', str(scontrol[0]), '-sy' , str(scontrol[1]), '-ln', '4', '-lp', '3', '--lncc', '7', '-pad', '0' , '-vel', '-voff'], stdout=subprocess.DEVNULL)
            subprocess.call([TRANSFORMcmd  , '-ref' , refFile , '-flow' , dummyFileNifti, nonlinearField], stdout=subprocess.DEVNULL)
            subprocess.call([REScmd , '-ref' , refMaskFile , '-flo', floMaskFile, '-trans' , nonlinearField , '-res' , outputMaskFile , '-inter', '0', '-voff'], stdout=subprocess.DEVNULL)

            #Saving images
            data = Image.open(outputFile)
            registered_image[it_batch:it_batch+1] = np.array(data) / 254.0

            data = Image.open(outputMaskFile)
            registered_mask[it_batch:it_batch+1] = np.array(data) / 254.0

            YY, XX = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')

            proxy = nib.load(nonlinearField)
            proxyarray = np.transpose(np.squeeze(np.asarray(proxy.dataobj)),[2, 1, 0])
            proxyarray[np.isnan(proxyarray)] = 0
            finalarray = np.zeros_like(proxyarray)
            finalarray[0] = proxyarray[0] - XX
            finalarray[1] = proxyarray[1] - YY
            velocity_field[it_batch] = finalarray

            flow_x = finalarray[0]/2**nstep
            flow_y = finalarray[1]/2**nstep
            for it_step in range(nstep):
                x = XX + flow_x
                y = YY + flow_y
                incx = bilinear_interpolate(flow_x, x, y)
                incy = bilinear_interpolate(flow_y, x, y)
                flow_x = flow_x + incx.reshape(image_shape)
                flow_y = flow_y + incy.reshape(image_shape)

            flow = np.concatenate((flow_x[np.newaxis],flow_y[np.newaxis]))
            displacement_field[it_batch] = flow

        else:
            registered_image[it_batch:it_batch + 1] = np.zeros(image_shape)
            registered_mask[it_batch:it_batch + 1] = np.zeros(image_shape)
            velocity_field[it_batch] = np.zeros((2,) + image_shape)
            displacement_field[it_batch] = np.zeros((2,) + image_shape)


    return np.transpose(registered_image,[1,2,0]), np.transpose(registered_mask,[1,2,0]), \
           np.transpose(velocity_field,[1,2,3,0]),  np.transpose(displacement_field,[1,2,3,0])

def integrate_NR(svf, image_shape):
    nstep = 7
    nslices = svf.shape[-1]

    YY, XX = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')

    flow_x = svf[0] / 2 ** nstep
    flow_y = svf[1] / 2 ** nstep

    flow = np.zeros((2,) + image_shape + (nslices,))
    for it_slice in range(nslices):
        fx = resize(flow_x[..., it_slice], image_shape)
        fy = resize(flow_y[..., it_slice], image_shape)
        if np.sum(fx) + np.sum(fy) == 0:
            continue

        for it_step in range(nstep):
            x = XX + fx
            y = YY + fy
            incx = bilinear_interpolate(fx, x, y)
            incy = bilinear_interpolate(fy, x, y)
            fx = fx + incx.reshape(image_shape)
            fy = fy + incy.reshape(image_shape)

        flow[0, ..., it_slice] = fx
        flow[1, ..., it_slice] = fy

    return flow

def integrate_RegNet(svf, image_shape, parameter_dict):

    nslices = svf.shape[-1]
    model = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=parameter_dict['IMAGE_SHAPE'],
        int_steps=7,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        rescaling=True,
        gaussian_filter_flag=True,
    )
    new_svf = np.zeros_like(svf)
    new_svf[0] = svf[1]
    new_svf[1] = svf[0]
    new_svf = torch.tensor(np.transpose(new_svf, [3, 0, 1, 2]))
    flow = model.get_flow_field(new_svf)
    flow = np.transpose(flow.detach().numpy(), [1, 2, 3, 0])

    flow_image = np.zeros((2,) + image_shape + (nslices,))
    transform = tf.Compose(parameter_dict['TRANSFORM'])
    for it_slice in range(nslices):
        f = flow[..., it_slice]
        f_y, f_x = transform.inverse([f[0], f[1]], img_shape=[image_shape] * 2)
        flow_image[0, ..., it_slice] = f_x
        flow_image[1, ..., it_slice] = f_y

    return flow_image

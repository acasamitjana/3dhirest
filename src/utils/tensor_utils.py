import itertools
import functools

import numpy as np
from torch.nn import init
import torch
import matplotlib.pyplot as plt

from src import models, layers
from src.utils import image_utils, visualization


class TensorDeformation(object):

    def __init__(self, image_shape, nonlinear_field_size, device):

        model = {}
        model['resize'] = models.ResizeTransform(inshape=nonlinear_field_size, target_size=image_shape, gaussian_filter_flag=False)
        model['vecInt'] = layers.VecInt(nonlinear_field_size)
        model['warper'] = layers.SpatialTransformer(image_shape)
        model['warper_affine'] = layers.SpatialTransformerAffine(image_shape)

        for mid, m in model.items():
            m.to(device)
            m.eval()

        self.model = model
        self.device = device

    def get_affine_matrix(self, angle):
        affine = torch.from_numpy(image_utils.get_affine_from_rotation(angle)).float().to(self.device)
        return affine

    def get_flow_field(self, low_res_nonfield):
        # low_res_deformation = self.model['vecInt'](low_res_nonfield)
        flow = self.model['resize'](low_res_nonfield)

        return flow

    def transform(self, image, angle=None, low_res_nonfield=None, flipud=0, fliplr=0, **kwargs):

        if fliplr > 0.5:
            image = torch.flip(image, [2])
        if flipud > 0.5:
            image = torch.flip(image, [3])

        if angle is not None:
            if any(angle):
                affine = torch.from_numpy(image_utils.get_affine_from_rotation(angle)).float().to(self.device)
                image = self.model['warper_affine'](image, affine, **kwargs)

        if low_res_nonfield is not None:
            # low_res_deformation = self.model['vecInt'](low_res_nonfield)
            deformation = self.model['resize'](low_res_nonfield)
            image = self.model['warper'](image, deformation, **kwargs)

        return image

    # def transform_inverse(self, image, angle=None, low_res_nonfield=None, flipud=0, fliplr=0, **kwargs):
    #
    #     if low_res_nonfield is not None:
    #         low_res_deformation = self.model['vecInt'](-low_res_nonfield)
    #         deformation = self.model['resize'](low_res_deformation)
    #         image = self.model['warper'](image, deformation, **kwargs)
    #
    #     if angle is not None:
    #         affine = torch.from_numpy(image_utils.get_affine_from_rotation(-angle)[np.newaxis]).float().to(self.device)
    #         image = self.model['warper_affine'](image, affine, **kwargs)
    #
    #     if fliplr > 0.5:
    #         image = torch.flip(image, [2])
    #     if flipud > 0.5:
    #         image = torch.flip(image, [3])
    #
    #     return image

class Init_net(object):


    def __init__(self):
        pass

    def init_net(self, net, init_type='normal', init_gain=0.02, device='cpu', gpu_ids=[]):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        Return an initialized network.
        """
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        self._init_weights(net, init_type, init_gain=init_gain)
        net = net.to(device)

        return net

    def _init_weights(self, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>

###################################
############ Functions ############
###################################

def flatten(v):
    """
    flatten Tensor v

    Parameters:
        v: Tensor to be flattened

    Returns:
        flat Tensor
    """

    return v.reshape(-1)

def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [torch.arange(0, d) for d in volshape]
    r, c = torch.meshgrid(*linvec)
    return (r,c)

def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [nb_features, *vol_shape]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'

    Returns:
        new interpolated volume of the same size as the entries in loc

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, dim=0)
    nb_dims = loc.shape[0]

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[1:])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = vol.expand((1,) + vol.shape)

    # flatten and float location Tensors
    loc = loc.type(torch.float32)

    volshape = vol.shape
    #
    # slices_2d = [loc[0], loc[1], vol[0]]
    # titles = ['loc', 'loc', 'vol']
    # slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True)

    # interpolate
    if interp_method == 'linear':
        loc0 = torch.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.shape[1:]]
        clipped_loc = [torch.clamp(loc[d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [torch.clamp(loc0[d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.type(torch.int32) for f in loc0lst], [f.type(torch.int32) for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.
        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(vol.shape[1:], subs)
            idx = idx.view([volshape[0],-1])

            vol_reshape = vol.reshape([volshape[0],-1])
            vol_val = torch.gather(vol_reshape, dim=1, index=idx).view(volshape[1:])
            # indices = torch.stack(subs, dim=0).type(torch.LongTensor)
            # print(indices[0].shape)
            # vol_val = vol[indices[0], indices[1]]

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            # wt = wt.expand((1,) + vol.shape)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest'
        roundloc = torch.round(loc).type('int32')

        # clip values
        max_loc = [(d - 1).type(torch.int32) for d in vol.shape]
        roundloc = [torch.clamp(roundloc[d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[1:], roundloc)
        interp_vol = torch.gather(vol.reshape([-1, vol.shape[-1]]), dim=0, index=idx)

    return interp_vol

def prod_n(lst):
    """
    Alternative to tf.stacking and prod, since tf.stacking can be slow
    """
    prod = lst[0].clone()
    for p in lst[1:]:
        prod *= p
    return prod

def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major
    """

    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1]).astype('int32')
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    ndx = ndx.type(torch.LongTensor)

    return ndx

def one_hot_encoding_tensor(target, num_classes, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (batch_size, 1, d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None:
        categories = list(range(num_classes))

    target = torch.squeeze(target)
    labels = torch.zeros((num_classes, ) + target.shape)
    for it_class in categories:
        idx_class = torch.where(target == it_class)
        idx = (it_class,) + idx_class
        labels[idx] = 1

    labels = torch.transpose(labels,1,0)

    return labels.float()

########################################
############ Transformation ############
########################################

def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in pytorch

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    This is based on (a.k.a. copied from) neuron code, so for more information and credit
    visit https://github.com/adalca/neuron/blob/master/neuron/utils.py

    """

    if affine_matrix.dtype != torch.float32:
        affine_matrix = affine_matrix.type(torch.float32)

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = affine_matrix.view(*[nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('RigidRegistration matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [f.type(torch.float32) for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(torch.ones(flat_mesh[0].shape, dtype=torch.float32))
    mesh_matrix = torch.transpose(torch.stack(flat_mesh, dim=1), dim0=0, dim1=1)  # 4 x nb_voxels

    # compute locations
    loc_matrix = torch.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = torch.transpose(loc_matrix[:nb_dims, :], dim0=0, dim1=1)  # nb_voxels x N
    loc = loc_matrix.reshape(list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - torch.stack(mesh, dim=nb_dims)


def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [nb_features, *vol_shape]
        loc_shift: shift volume [N, *new_vol_shape]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """


    # parse shapes
    volshape = loc_shift.shape[1:]
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [mesh[d].type(torch.float32) + loc_shift[d] for d in range(nb_dims)]

    # slices_2d = [loc[0], loc_shift[0], loc[1], loc_shift[1]]
    # titles = ['mesh', 'loc_shift', 'mesh', 'loc_shift']
    # slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True)

    # test single
    return interpn(vol, loc, interp_method=interp_method)


def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """
    Integrate (stationary of time-dependent) vector field (N-D Tensor) in tensorflow

    Aside from directly using tensorflow's numerical integration odeint(), also implements
    "scaling and squaring", and quadrature. Note that the diff. equation given to odeint
    is the one used in quadrature.

    Parameters:
        vec: the Tensor field to integrate.
            If vol_size is the size of the intrinsic volume, and vol_ndim = len(vol_size),
            then vector shape (vec_shape) should be
            [vol_size, vol_ndim] (if stationary)
            [vol_size, vol_ndim, nb_time_steps] (if time dependent)
        time_dep: bool whether vector is time dependent
        method: 'scaling_and_squaring' or 'ss' or 'ode' or 'quadrature'

        if using 'scaling_and_squaring': currently only supports integrating to time point 1.
            nb_steps: int number of steps. Note that this means the vec field gets broken
            down to 2**nb_steps. so nb_steps of 0 means integral = vec.

        if using 'ode':
            out_time_pt (optional): a time point or list of time points at which to evaluate
                Default: 1
            init (optional): if using 'ode', the initialization method.
                Currently only supporting 'zero'. Default: 'zero'
            ode_args (optional): dictionary of all other parameters for
                tf.contrib.integrate.odeint()

    Returns:
        int_vec: integral of vector field.
        Same shape as the input if method is 'scaling_and_squaring', 'ss', 'quadrature',
        or 'ode' with out_time_pt not a list. Will have shape [*vec_shape, len(out_time_pt)]
        if method is 'ode' with out_time_pt being a list.

    Todo:
        quadrature for more than just intrinsically out_time_pt = 1
    """

    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            raise ValueError("Method=" + method + " and time_dependent not yet implemented")
        else:
            disp = vec.clone()
            disp = disp / (2 ** nb_steps)
            for _ in range(nb_steps):
                disp += transform(disp, disp)
            disp = vec

    elif method == 'quadrature':
        raise ValueError("Method=" + method + " not yet implemented")

    else:
        raise ValueError("Method=" + method + " not yet implemented")


    return disp

import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import convolve

def build_regular_space_volume(slices, z_pos, cog_xy, res_xy, target_sp=0.5, max_z_error=0.1):

    z1 = np.floor(min(z_pos))
    z2 = np.ceil(max(z_pos))
    z = np.arange(z1, z2+1, step=target_sp)
    nz = len(z)

    vol = np.zeros([*slices.shape[:-1], nz])
    for i in range(nz):
        dists = np.abs(z_pos - z[i])
        j = np.argmin(dists).astype('int')
        if dists[j] < max_z_error:
            vol[:,:,i] = slices[:,:,j]

    aff = np.zeros([4, 4])
    aff[0,0] = res_xy
    aff[1,1] = res_xy
    aff[2,2] = target_sp
    aff[3,3] = 1
    aff[0,-1] = -cog_xy[0]
    aff[1,-1] = -cog_xy[1]
    aff[2,-1] = z1

    return vol, aff

def build_regular_space_volume_color(slices, z_pos, cog_xy, res_xy, target_sp=0.5, max_z_error=0.1):

    z1 = np.floor(min(z_pos))
    z2 = np.ceil(max(z_pos))
    z = np.arange(z1, z2+1, step=target_sp)
    nz = len(z)

    vol = np.zeros([*slices.shape[:-2], nz, 3])
    for i in range(nz):
        dists = np.abs(z_pos - z[i])
        j = np.argmin(dists).astype('int')
        if dists[j] < max_z_error:
            vol[:,:,i, :] = slices[:,:,j, :]

    aff = np.zeros([4, 4])
    aff[0,0] = res_xy
    aff[1,1] = res_xy
    aff[2,2] = target_sp
    aff[3,3] = 1
    aff[0,-1] = -cog_xy[0]
    aff[1,-1] = -cog_xy[1]
    aff[2,-1] = z1

    return vol, aff

def one_hot_encoding(target, num_classes, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None:
        categories = list(range(num_classes))

    labels = np.zeros((num_classes,) + target.shape)
    for it_class in categories:
        idx_class = np.where(target == it_class)
        idx = (it_class,)+ idx_class
        labels[idx] = 1

    return labels.astype(int)

def get_affine_from_rotation(angle_list):

    affine_matrix = np.zeros((len(angle_list), 2,3))
    for it_a, angle in enumerate(angle_list):
        angle_rad = angle * np.pi / 180
        affine_matrix[it_a] = np.array([
            [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
            [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
        ])
    return affine_matrix

def grad3d(x):

    filter = np.asarray([-1,0,1])
    gx = convolve(x, np.reshape(filter, (3,1,1)), mode='constant')
    gy = convolve(x, np.reshape(filter, (1,3,1)), mode='constant')
    gz = convolve(x, np.reshape(filter, (1,1,3)), mode='constant')

    gx[0], gx[-1] = x[1] - x[0], x[-1] - x[-2]
    gy[:, 0], gy[:, -1] = x[:,1] - x[:,0], x[:, -1] - x[:, -2]
    gz[..., 0], gz[..., -1] = x[..., 1] - x[..., 0], x[..., -1] - x[..., -2]

    gmodule = np.sqrt(gx**2 + gy**2 + gz**2)
    return gmodule, gx, gy, gz

def bilinear_interpolate(im, x, y):
    '''

    :param im: 2D image with size NxM
    :param x: coordinates in 'x' (M columns of a matrix)
    :param y: coordinates in 'y' (N rows of a matrix)
    :return:
    '''

    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def bilinear_interpolate3d(im, x, y, z):
    '''

    :param im: 2D image with size NxM
    :param x: coordinates in 'x' (M columns of a matrix)
    :param y: coordinates in 'y' (N rows of a matrix)
    :param z: coordinates in 'z' (L)
:return:
    '''

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)
    z0 = np.clip(z0, 0, im.shape[2] - 1)
    z1 = np.clip(z1, 0, im.shape[2] - 1)

    Ia = im[y0, x0, z0]
    Ib = im[y1, x0, z0]
    Ic = im[y0, x1, z0]
    Id = im[y1, x1, z0]
    Ie = im[y0, x0, z1]
    If = im[y1, x0, z1]
    Ig = im[y0, x1, z1]
    Ih = im[y1, x1, z1]

    wa = (x1 - x) * (y1 - y) * (z1 - z)
    wb = (x1 - x) * (y - y0) * (z1 - z)
    wc = (x - x0) * (y1 - y) * (z1 - z)
    wd = (x - x0) * (y - y0) * (z1 - z)
    we = (x1 - x) * (y1 - y) * (z0 - z)
    wf = (x1 - x) * (y - y0) * (z0 - z)
    wg = (x - x0) * (y1 - y) * (z0 - z)
    wh = (x - x0) * (y - y0) * (z0 - z)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih

def deform2D(image, field, mode='bilinear'):
    '''

    :param image: 2D np.array (nrow, ncol)
    :param field: 3D np.array (2, nrow, ncol)
    :param mode: 'bilinear' or 'nearest'
    :return:
    '''

    dx = field[0]
    dy = field[1]
    output_shape = field.shape[1:]
    if len(image.shape) > 2: #RGB
        output = np.zeros(output_shape + (3,))
        YY, XX = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')
        XXd = XX + dx
        YYd = YY + dy
        for it_c in range(3):
            if mode == 'bilinear':
                output[:,:,it_c] = bilinear_interpolate(image[:,:,it_c], XXd, YYd)
            elif mode == 'nearest':
                output[:,:,it_c] = griddata((YY.flatten(), XX.flatten()), image[:,:,it_c].flatten(), (YYd, XXd), method='nearest')
            else:
                raise ValueError('Interpolation mode not available')
    else:
        YY, XX = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')

        XXd = XX+dx
        YYd = YY+dy
        if mode == 'bilinear':
            output = bilinear_interpolate(image, XXd, YYd)
        elif mode == 'nearest':
            output = griddata((YY.flatten(), XX.flatten()), image.flatten(), (YYd, XXd), method='nearest')
        else:
            raise ValueError('Interpolation mode not available')


    return output

def affine_to_dense(affine_matrix, volshape):

    ndims = len(volshape)

    vectors = [np.arange(0, s) for s in volshape]
    mesh = np.meshgrid(*vectors, indexing=('ij')) #grid of vectors
    mesh = [f.astype('float32') for f in mesh]
    mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(ndims)] #shift center

    # mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    # mesh = [tf.cast(f, 'float32') for f in mesh]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [np.reshape(f, (-1,)) for f in mesh]
    flat_mesh.append(np.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = np.transpose(np.stack(flat_mesh, axis=1))  # ndims+1 x nb_voxels

    # compute locations
    loc_matrix = np.matmul(affine_matrix, mesh_matrix)  # ndims+1 x nb_voxels
    loc = np.reshape(loc_matrix[:ndims, :], [ndims] + list(volshape))  # ndims x *volshape

    # get shifts and return

    shift = loc - np.stack(mesh, axis=0)
    return shift.astype('float32')

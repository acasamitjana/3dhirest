import copy

from skimage.transform import resize
from scipy.interpolate import griddata, bisplev, bisplrep
import numpy as np

from src.utils.image_utils import bilinear_interpolate, deform2D


##################################################
##################  Parameters  ##################
##################################################
class ResizeParams(object):
    def __init__(self, resize_shape):
        if isinstance(resize_shape, int):
            resize_shape = (resize_shape, resize_shape)
        self.resize_shape = resize_shape

class CropParams(object):
    def __init__(self, crop_shape, init_coordinates = None):
        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape)
        self.crop_shape = crop_shape
        self.init_coordinates = init_coordinates

class AffineParams(object):
    def __init__(self, rotation, x_translation, y_translation):
        self.x_translation = x_translation
        self.y_translation = y_translation
        self.rotation = rotation

class FlipParams(object):
    pass

class PadParams(object):
    def __init__(self, psize, pfill=0, pmode='constant', dim=2):
        if isinstance(psize, int):
            psize = (psize, psize)
        self.psize = psize
        self.pmode = pmode
        self.pfill = pfill
        self.dim = dim

class NonLinearParams(object):
    def __init__(self, lowres_size, lowres_strength=1, distribution = 'normal', nstep=5):
        self.lowres_size = lowres_size
        self.lowres_strength = lowres_strength
        self.distribution = distribution
        self.nstep = nstep

class RotationParams(object):

    def __init__(self, value_range, distribution='normal'):
        self.value_range = value_range
        self.distribution = distribution

##################################################
###################  Functions  ##################
##################################################

class Compose(object):
    def __init__(self, transform_parameters):

        self.transform_parameters = transform_parameters if transform_parameters is not None else []
        self.img_shape = None

    def _compute_data_shape(self, init_shape):

        if isinstance(init_shape, list):
            n_shape = len(init_shape)
            final_shape = init_shape

        else:
            n_shape = 1
            final_shape = [init_shape]

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                final_shape = [t.crop_shape] * n_shape

            elif isinstance(t, PadParams):
                if t.psize is None:
                    final_shape = init_shape
                    #     psize = max([max([di.size for di in d]) for d in self.data])
                    #     t.psize = (1 << (psize[0] - 1).bit_length(), 1 << (psize[1] - 1).bit_length())
                else:
                    final_shape = [t.psize] * n_shape

            elif isinstance(t, ResizeParams):
                final_shape = [t.resize_shape] * n_shape

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        if isinstance(init_shape, list):
            return final_shape
        else:
            return final_shape[0]

    def __call__(self, img):

        img_shape = [i.shape for i in img]

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                tf = RandomCropManyImages(t)
                img = tf(img)

            elif isinstance(t, PadParams):
                img = [Padding(t, i.shape)(i) for i in img]

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        self.img_shape = img_shape

        return img

    def inverse(self, img, img_shape=None):

        if img_shape is None:
            if self.img_shape is None:
                raise ValueError("You need to provide the initial image shape or call the forward transform function"
                                 "before calling the inverse")
            else:
                img_shape = self.img_shape


        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                tf = RandomCropManyImages(t)
                img = tf.inverse(img, img_shape)

            elif isinstance(t, PadParams):
                img = [Padding(t, i.shape).inverse(i, img_shape) for i in img]

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        return img

class Compose_DA(object):
    def __init__(self, data_augmentation_parameters):
        self.data_augmentation_parameters = data_augmentation_parameters if data_augmentation_parameters is not None else []

    def __call__(self, img, mask_flag = None, **kwargs):
        '''
        Mask flag is used to indicate which elements of the list are not used in intensity-based transformations, and
        only in deformation-based transformations.
        '''

        islist = True
        if not isinstance(img, list):
            img = [img]
            islist = False

        if mask_flag is None:
            mask_flag = [False] * len(img)
        elif not isinstance(mask_flag, list):
            mask_flag = [mask_flag] * len(img)


        for da in self.data_augmentation_parameters:

            if isinstance(da, NonLinearParams):
                tf = NonLinearDifferomorphismManyImages(da)
                img = tf(img, mask_flag)

            elif isinstance(da, RotationParams):
                tf = Rotation(da)
                img, flow = tf(img, mask_flag)

            else:
                raise ValueError(str(type(da)) + 'is not a valid type for data augmentation. Please, specify a valid one')

        if not islist:
            img = img[0]

        return img

class NormalNormalization(object):
    def __init__(self, mean = 0, std = 1, dim = None, inplace = False):

        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.dim = None

    def __call__(self, data, *args, **kwargs):
        if not self.inplace:
            data = copy.deepcopy(data)

        mean_d = np.mean(data, axis = self.dim)
        std_d = np.std(data, axis = self.dim)

        assert len(mean_d) == self.mean

        d_norm = (data - mean_d) / std_d
        out_data = (d_norm + self.mean) * self.std

        return out_data

class ScaleNormalization(object):
    def __init__(self, scale=1.0, range = None, percentile=None):

        self.scale = scale
        self.range = range
        self.percentile = percentile

    def __call__(self, data, *args, **kwargs):

        if self.range is not None:
            if self.percentile is not None:
                dmax = np.percentile(data,self.percentile)
                dmin = np.percentile(data,100-self.percentile)

            else:
                dmax = np.max(data)
                dmin = np.min(data)

            if dmax != dmin:
                data = (data - dmin) / (dmax-dmin) * (self.range[1] - self.range[0]) + self.range[0]
        else:
            data = data * self.scale

        return data

class Padding(object):
    def __init__(self, parameters, isize, dim=2):


        if len(isize) > dim+1:
            raise ValueError("Please, specify a valid dimension and size")

        osize = parameters.psize
        assert len(osize) == dim

        pfill = parameters.pfill
        pmode = parameters.pmode

        psize = []
        for i, o in zip(isize, osize):
            if o - i > 0:
                pfloor = int(np.floor((o - i) / 2))
                pceil = pfloor if np.mod(o - i, 2) == 0 else pfloor + 1
            else:
                pfloor = 0
                pceil = 0

            psize.append((pfloor, pceil))

        pad_tuple = psize

        self.padding = pad_tuple
        self.fill = pfill
        self.padding_mode = pmode
        self.dim = dim
        self.osize = osize

    def __call__(self, data):

        if len(data.shape) == self.dim+1:
            nchannels = data.shape[-1]
            output_data = np.zeros(self.osize + (nchannels, ))
            for idim in range(nchannels):
                output_data[..., idim] = np.pad(data[..., idim], pad_width=self.padding, mode=self.padding_mode,
                                                constant_values=self.fill)
            return output_data
        else:
            return np.pad(data, pad_width=self.padding, mode=self.padding_mode, constant_values=self.fill)

class RandomCropManyImages(object):
    """Crop the given numpy array at a random location.
    Images are cropped at from the center as follows:


    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (d1, d2, ... , dN), a square crop (size, size, ..., size) is
            made.

        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding.
    """

    def __init__(self, parameters, pad_if_needed=True, fill=0, padding_mode='constant'):

        self.parameters = parameters
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def get_params(self, data_shape, output_shape):

        if all([a==b for a,b in zip(data_shape, output_shape)]):
            return [0]*len(data_shape), data_shape

        if self.parameters.init_coordinates is None:
            init_coordinates = []
            for a, b in zip(data_shape, output_shape):
                init_coordinates.append(int((a-b)/2))
        else:
            init_coordinates = self.parameters.init_coordinates

        return init_coordinates, output_shape

    def __call__(self, data_list):
        """
        Args:
            data_list : list of numpy arrays. Each numpy array has the following size: (num_channels, d1, ..., dN)

        Returns:
            output_list: list of cropped numpy arrays.
        """
        size = self.parameters.crop_shape
        n_dims = len(size)
        padded_data_list = []
        for i in range(len(data_list)):
            data = data_list[i]
            # pad the width if needed
            pad_width = []
            for it_dim in range(n_dims):
                if self.pad_if_needed and data.shape[it_dim] < size[it_dim]:
                    pad_width.append((size[it_dim] - data.shape[it_dim],0))
                else:
                    pad_width.append((0,0))

            data = np.pad(data, pad_width=pad_width, mode=self.padding_mode, constant_values=self.fill)
            padded_data_list.append(data)


        init_coord, output_shape = self.get_params(padded_data_list[0].shape, size)

        self.init_coord = init_coord
        self.output_shape = output_shape

        output = []
        for i in range(len(padded_data_list)):
            padded_data = padded_data_list[i]
            for it_dim in range(n_dims):
                idx = (slice(None),) * (it_dim) + \
                      (slice(init_coord[it_dim], init_coord[it_dim] + output_shape[it_dim], 1), )
                padded_data = padded_data[idx]
            output.append(padded_data)

        return output

    def inverse(self, data_list, data_shape):
        size = self.parameters.crop_shape
        n_dims = len(size)

        cropped_data_list = []
        for data, dshape in zip(data_list, data_shape):
            cropped_data = data
            for it_dim in range(n_dims):
                init_coord = size[it_dim] - dshape[it_dim]
                if init_coord < 0:
                    init_coord = 0

                idx = (slice(None),) * (it_dim) + (slice(init_coord, init_coord + dshape[it_dim], 1),)
                cropped_data = cropped_data[idx]
            cropped_data_list.append(cropped_data)

        init_coord, _ = self.get_params(data_shape[0], size)

        output = []
        for data, dshape in zip(cropped_data_list, data_shape):
            pad_width = []
            for it_dim in range(n_dims):
                if size[it_dim] < dshape[it_dim]:
                    pad_width.append((int(init_coord[it_dim]), int(dshape[it_dim] - size[it_dim] - init_coord[it_dim])))
                else:
                    pad_width.append((0, 0))

            data = np.pad(data, pad_width=pad_width, mode=self.padding_mode, constant_values=self.fill)
            output.append(data)

        return output


class NonLinearDeformationManyImages(object):

    def __init__(self, params, output_flow=False, reverse_field=False):
        self.params = params
        self.output_flow = output_flow
        self.reverse_field = reverse_field

    def _get_lowres_strength(self,):

        size = 1
        if self.params.distribution == 'normal':
            mean, std = self.params.lowres_strength[1], self.params.lowres_strength[0]
            lowres_strength = np.random.randn(size) * std + mean

        elif self.params.distribution == 'uniform':
            high, low = self.params.lowres_strength[1], self.params.lowres_strength[0]
            lowres_strength = np.random.rand(size) * (high - low) + low

        elif self.params.distribution == 'lognormal':
            mean, std = self.params.lowres_strength[1], self.params.lowres_strength[0]
            lowres_strength = np.random.randn(size) * std + mean
            lowres_strength = np.exp(lowres_strength)

        elif self.params.distribution is None:
            lowres_strength = [self.params.lowres_strength] * size

        else:
            raise ValueError("[src/utils/transformations: NonLinearDeformation]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        field_lowres_x = lowres_strength * np.random.randn(self.params.lowres_size[0],
                                                           self.params.lowres_size[1])  # generate random noise.

        field_lowres_y = lowres_strength * np.random.randn(self.params.lowres_size[0],
                                                           self.params.lowres_size[1])  # generate random noise.

        return field_lowres_x, field_lowres_y

    def __call__(self, data, mask_flag, XX, YY, flow_x, flow_y, *args, **kwargs):

        x, y = XX + flow_x, YY + flow_y

        data_tf = []
        for it_image, (image, m) in enumerate(zip(data, mask_flag)):
            if m:
                data_tf.append(griddata((YY.flatten(), XX.flatten()), image.flatten(), (y, x), method='nearest'))
            else:
                # data_tf.append(np.double(bilinear_interpolate(image, y, x)>0.5))
                data_tf.append(bilinear_interpolate(image, x, y))
        return data_tf

class NonLinearDifferomorphismManyImages(NonLinearDeformationManyImages):


    def get_diffeomorphism(self, field_lowres_x, field_lowres_y, image_shape, reverse=False):

        field_highres_x = resize(field_lowres_x, image_shape)
        field_highres_y = resize(field_lowres_y, image_shape)

        # integrate
        YY, XX = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')

        flow_x = field_highres_x / (2 ** self.params.nstep)
        flow_y = field_highres_y / (2 ** self.params.nstep)

        if reverse:
            flow_x = -flow_x
            flow_y = -flow_y

        for it_step in range(self.params.nstep):
            x = XX + flow_x
            y = YY + flow_y
            incx = bilinear_interpolate(flow_x, x, y)
            incy = bilinear_interpolate(flow_y, x, y)
            flow_x = flow_x + incx.reshape(image_shape)
            flow_y = flow_y + incy.reshape(image_shape)

        return XX, YY, flow_x, flow_y

    def __call__(self, data, mask_flag, *args, **kwargs):

        image_shape = data[0].shape
        field_lowres_x, field_lowres_y = self._get_lowres_strength()

        XX, YY, flow_x, flow_y = self.get_diffeomorphism(field_lowres_x, field_lowres_y, image_shape)
        data_tf = super().__call__(data, mask_flag, XX, YY, flow_x, flow_y)

        if self.output_flow:
            if self.reverse_field:
                XX, YY, flow_x, flow_y = self.get_diffeomorphism(field_lowres_x, field_lowres_y, image_shape, reverse=True)

            return data_tf, np.stack([flow_x, flow_y], axis=0)
        else:
            return data_tf

class NonLinearBSplinesManyImages(NonLinearDeformationManyImages):

    def get_deformation_field(self, field_lowres_x, field_lowres_y, image_shape):
        '''

        :param field_lowres_x: np.array of low resolution velocity field (possibly random)
        :param field_lowres_y: np.array low resolution velocity field (possibly random)
        :param image_shape: tuple of (rows, cols)
        :return: flow_x, flow_y

        I'll be using X and Y as cartesian coordinates (i.e: Y=rows, X=columns). However, b-Splines functions use
        X=rows, Y=columns, that's why I change the parameters
        '''
        lowres_shape = field_lowres_x.shape
        ratio_shape = tuple([np.ceil(1.0*a/b) for a,b in zip(image_shape,lowres_shape)])

        Y, X = np.arange(0, image_shape[0]), np.arange(0, image_shape[1])
        Ylowres, Xlowres = np.arange(0, image_shape[0],ratio_shape[0]), np.arange(0, image_shape[1],ratio_shape[1])
        YYlowres, XXlowres = np.meshgrid(Ylowres, Xlowres, indexing='ij')

        #bisplrep uses x as rows and y as columns
        r_x = bisplrep(YYlowres, XXlowres, field_lowres_x.flatten(), tx=np.arange(0, Ylowres.shape[0]), ty=np.arange(0, Xlowres.shape[0]), task=-1)#, s=np.prod(Xlowres.shape))
        r_y = bisplrep(YYlowres, XXlowres, field_lowres_y.flatten(), tx=np.arange(0, Ylowres.shape[0]), ty=np.arange(0, Xlowres.shape[0]), task=-1)#, s=np.prod(Xlowres.shape))

        flow_x = bisplev(Y, X, r_x)
        flow_y = bisplev(Y, X, r_y)

        # (X=rows,Y=col)
        return flow_x, flow_y

    def __call__(self, data, mask_flag, *args, **kwargs):

        image_shape = data[0].shape
        field_lowres_x, field_lowres_y = self._get_lowres_strength()

        flow_x, flow_y = self.get_deformation_field(field_lowres_x, field_lowres_y, image_shape)
        YY, XX = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')
        data_tf = super().__call__(data, mask_flag, XX, YY, flow_x, flow_y)

        if self.output_flow:
            return data_tf, np.stack([flow_x, flow_y], axis=0)
        else:
            return data_tf

class Rotation(object):
    def __init__(self, params, dense_field=False, reverse=True):
        '''

        :param params: instance of RotationParams
        :param dense_field: affine matrix transformation to dense field.
        :param reverse: it outputs the reverse transformation
        '''
        self.params = params
        self.dense_field = dense_field
        self.reverse = reverse

    def _get_angle(self, size):

        if self.params.distribution == 'normal':
            mean, std = self.params.value_range[1], self.params.value_range[0]
            angle = np.random.randn(size) * std + mean

        elif self.params.distribution == 'uniform':
            high, low = self.params.value_range[1], self.params.value_range[0]
            angle = np.random.rand(size) * (high - low) + low

        elif self.params.distribution is None:
            angle = self.params.lowres_strength * np.ones(size)

        else:
            raise ValueError("[src/utils/transformations: MultiplicativeNoise]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        return angle

    def _get_affine_matrix(self, angle):
        angle_rad = angle * np.pi / 180
        affine_matrix = np.array([
            [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
            [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
            [0, 0, 1]
        ])
        return affine_matrix

    def _get_dense_field(self, affine_matrix, volshape):

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
        shift = np.stack([shift[1], shift[0]], axis=0)
        return shift.astype('float32')

    def __call__(self, data, mask_flag, *args, **kwargs):
        '''
        :param data: 2D data
        :param mask_flag: True = nearest interpolation, False = bilinear interpolation
        :return:
        '''

        angle = self._get_angle(1)
        affine_matrix = self._get_affine_matrix(angle)
        flow = self._get_dense_field(affine_matrix, data[0].shape)
        data_tf = []
        for image, m in zip(data, mask_flag):
            o = 'nearest' if m else 'bilinear'
            data_tf.append(deform2D(image,flow,o))

        if self.dense_field:
            if self.reverse:
                affine_matrix = self._get_affine_matrix(-angle)
                flow = self._get_dense_field(affine_matrix, data[0].shape)
            return data_tf, flow
        else:
            if self.reverse:
                affine_matrix = self._get_affine_matrix(-angle)
            return data_tf, affine_matrix

class AffineTransformation(object):
    def __init__(self, affine_matrix):
        self.affine_matrix = affine_matrix

    def _get_dense_field(self, volshape):

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
        mesh_matrix = np.transpose(np.stack(flat_mesh, axis=1))  # 4 x nb_voxels
        # compute locations
        loc_matrix = np.matmul(self.affine_matrix, mesh_matrix)  # N+1 x nb_voxels
        # loc_matrix = np.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
        loc = np.reshape(loc_matrix[:ndims, :], [ndims] + list(volshape))  # *volshape x N
        # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

        # get shifts and return

        shift = loc - np.stack(mesh, axis=0)
        return loc.astype('float32'), shift.astype('float32')

    def __call__(self, data, mask_flag, *args, **kwargs):
        image_shape = data.shape
        XX, YY = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing=('ij'))

        new_loc, flow = self._get_dense_field(image_shape)
        flow_x, flow_y = flow[0], flow[1]

        x = XX + flow_x
        y = YY + flow_y

        if mask_flag:
            data_tf = griddata((XX.flatten(), YY.flatten()), data.flatten(), (x, y), method='nearest')
        else:
            # data_tf.append(np.double(bilinear_interpolate(image, y, x)>0.5))
            data_tf = bilinear_interpolate(data, y, x)

        return data_tf




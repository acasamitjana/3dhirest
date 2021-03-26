import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.visualization import slices
# Define a ResNet block

#########################################
############ Learning layers ############
#########################################

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm_layer == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm_layer == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class BaseConvBlock2D(nn.Module):

    def initialize_padding(self, pad_type, padding):
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

    def initialize_normalization(self, norm_layer, norm_dim):
        if norm_layer == 'bn':
            self.norm_layer = nn.BatchNorm2d(norm_dim)
        elif norm_layer == 'in':
            self.norm_layer = nn.InstanceNorm2d(norm_dim, affine=False)
        elif norm_layer == 'ln':
            self.norm_layer = LayerNorm(norm_dim)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm_layer = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

    def initialize_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

class ConvBlock2D(BaseConvBlock2D):
    '''
    2D ConvlutionBlock performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param Conv2D input parameters: see nn.Conv2D
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_size=3, padding=0, stride=1, bias=True,
                 norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        # initialize padding
        self.initialize_padding(pad_type, padding)
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)
        self.conv_layer = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size,  stride=stride, bias=bias)


    def forward(self, inputs):
        outputs = self.conv_layer(self.pad(inputs))
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class ConvTransposeBlock2D(BaseConvBlock2D):
    '''
    2D ConvTransposeBlock2D performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param ConvTranspose2D input parameters: see nn.ConvTranspose2d
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_sizeT=4, kernel_size=3, output_padding=0, padding=0,
                 stride=2, bias=True, norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        self.initialize_padding(pad_type, padding, int(np.floor((kernel_size-1)/2)))
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)

        self.convT_layer = nn.ConvTranspose2d(input_filters, input_filters, kernel_size=kernel_sizeT,
                                              output_padding=output_padding, stride=stride, bias=bias)

        self.conv_layer = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size,
                                    stride=1, bias=bias)


    def initialize_padding(self, pad_type, padding1, padding2):
        # initialize padding
        if pad_type == 'reflect':
            self.pad1 = nn.ReflectionPad2d(padding1)
            self.pad2 = nn.ReflectionPad2d(padding2)

        elif pad_type == 'replicate':
            self.pad1 = nn.ReplicationPad2d(padding1)
            self.pad2 = nn.ReplicationPad2d(padding2)

        elif pad_type == 'zeros':
            self.pad1 = nn.ZeroPad2d(padding1)
            self.pad2 = nn.ZeroPad2d(padding2)

        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)


    def forward(self, inputs):
        outputs = self.convT_layer(self.pad1(inputs))
        outputs = self.conv_layer(self.pad2(outputs))

        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class ResnetBlock(nn.Module):

    def __init__(self, num_filters, padding_type='zeros', norm_layer='bn', use_dropout=False,
                 use_bias=False, activation='relu'):

        super().__init__()
        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block.append(ConvBlock2D(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias,
                                      norm_layer=norm_layer, activation=activation, pad_type=padding_type))

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block.append(ConvBlock2D(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias,
                                      norm_layer=norm_layer, activation=activation, pad_type=padding_type))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual

        return out

class ResBlock(nn.Module):

    def __init__(self, num_filters, pad_type='reflect', norm_layer='bn',use_bias=True, activation='relu'):

        super().__init__()
        conv_block = []
        conv_block += [ConvBlock2D(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias,
                                   norm_layer=norm_layer, activation=activation, pad_type=pad_type)]

        conv_block += [ConvBlock2D(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias,
                                   norm_layer=norm_layer, activation='none', pad_type=pad_type)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual

        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm_layer='in', activation='relu', pad_type='zeros'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm_layer=norm_layer, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class UpConv(nn.Module):

    def __init__(self, input_filters, output_filters, kernel_size=3, stride=2,
                 bias=True, norm_layer='bn', activation='relu', mode='bilinear', pad_type='zeros'):

        super().__init__()
        self.up_layer = nn.Upsample(scale_factor=stride, mode=mode)

        self.conv_layer = ConvBlock2D(input_filters, output_filters, kernel_size=kernel_size, padding=1, stride=1,
                                      norm_layer=norm_layer, activation=activation, bias=bias, pad_type=pad_type)


    def forward(self, inputs):
        outputs = self.up_layer(inputs)
        outputs =  self.conv_layer(outputs)


        return outputs



###############################################
############ Transformation layers ############
###############################################
class AffineTransformer(nn.Module):
    def __init__(self, vol_shape, input_channels, enc_features):
        super(AffineTransformer, self).__init__()

        # Spatial transformer localization-network
        out_shape = [v for v in vol_shape]
        nf_list = [input_channels] + enc_features
        localization_layers = []
        for in_nf, out_nf in zip(nf_list[:-1], nf_list[1:]):
            localization_layers.append(nn.Conv2d(in_nf, out_nf, kernel_size=3, stride=2, padding=1))
            localization_layers.append(nn.LeakyReLU(0.2))
            out_shape = [o/2 for o in out_shape]

        self.localization = nn.Sequential(*localization_layers)
        self.out_shape = int(enc_features[-1]*np.prod(out_shape))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.out_shape, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 3 * 2)
        )

    # Spatial transformer network forward function
    def forward(self, x):
        x_floating = x[:,0:1]
        xs = self.localization(x)
        xs = xs.view(-1, self.out_shape)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x_floating.size())

        return F.grid_sample(x_floating, grid), theta

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode = 'border'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode
        self.padding_mode = padding_mode


    def forward(self, src, flow, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image of size [batch_size, n_dims, *volshape]
            :param flow: the output from the U-Net [batch_size, n_dims, *volshape]
        """

        if 'mode' in kwargs:
            self.mode = kwargs['mode']

        new_locs = self.grid + flow

        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
        else:
            shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode=self.padding_mode)

class SpatialTransformerAffine(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode = 'border', torch_dtype=torch.float):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        ndims = len(size)

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = grid.type(torch_dtype)


        flat_mesh = torch.reshape(grid, (ndims,-1))
        ones_vec = torch.ones((1, np.prod(size))).type(torch_dtype)
        mesh_matrix = torch.cat((flat_mesh, ones_vec), dim=0)

        # grid = torch.unsqueeze(grid, 0)  # add batch
        # grid = grid.type(torch_dtype)
        # self.register_buffer('grid', grid)

        mesh_matrix = mesh_matrix.type(torch_dtype)
        self.register_buffer('mesh_matrix', mesh_matrix)

        self.size = size
        self.mode = mode
        self.padding_mode = padding_mode
        self.torch_dtype = torch_dtype

    def _get_locations(self, affine_matrix):

        batch_size = affine_matrix.shape[0]
        ndims = len(self.size)
        vol_shape = self.size

        # compute locations
        loc_matrix = torch.matmul(affine_matrix, self.mesh_matrix)  # N x nb_voxels
        loc = torch.reshape(loc_matrix, [batch_size, ndims] + list(vol_shape))  # *volshape x N

        return loc.float()

    def forward(self, src, affine_matrix, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image of size [batch_size, n_dims, *volshape]
            :param flow: the output from the U-Net [batch_size, n_dims, *volshape]
        """

        if 'mode' in kwargs:
            self.mode = kwargs['mode']

        affine_matrix = affine_matrix.type(self.torch_dtype)

        new_locs = self._get_locations(affine_matrix)
        new_locs = new_locs.type(self.torch_dtype)

        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
        else:
            shape = new_locs.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]


        return F.grid_sample(src, new_locs, mode='bilinear', padding_mode=self.padding_mode)

class VecInt(nn.Module):
    """
    Vector Integration Layer

    Enables vector integration via several methods
    (ode or quadrature for time-dependent vector fields,
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, field_shape, int_steps=7, **kwargs):
        """
        Parameters:
            int_steps is the number of integration steps
        """
        super().__init__()
        self.int_steps = int_steps
        self.scale = 1 / (2 ** self.int_steps)
        self.transformer = SpatialTransformer(field_shape)

    def forward(self, field, **kwargs):

        output = field
        output = output * self.scale
        nsteps = self.int_steps
        if 'nsteps' in kwargs:
            nsteps = nsteps - kwargs['nsteps']

        for _ in range(nsteps):
            a = self.transformer(output, output)
            output = output + a

        return output

class RescaleTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, factor=None, target_size=None, gaussian_filter_flag=True):
        '''

        :param vol_size:
        :param factor:
                :param latent_size: it only applies if factor is None

        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag

        if factor is None:
            assert target_size is not None
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])
        elif isinstance(factor, list) or isinstance(factor, tuple):
            self.factor = list(factor)
        else:
            self.factor = [factor for _ in range(self.ndims)]

        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if self.factor[0] < 1 and self.gaussian_filter_flag:

            kernel_sigma = [0.44 * f for f in self.factor]
            kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)

            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x):

        # x = x.clone()
        if self.factor[0] < 1:
            if self.gaussian_filter_flag:
                padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
                x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)

            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]

        elif self.factor[0] > 1:
            # multiply first to save memory
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, target_size=None, factor=None, gaussian_filter_flag=True):
        '''

        :param vol_size:
        :param factor: if factor<1 the shape is reduced and viceversa.
        :param latent_size: it only applies if factor is None
        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag
        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if target_size is None:
            self.factor = factor
            if isinstance(factor, float) or isinstance(factor, int):
                self.factor = [factor for _ in range(self.ndims)]
        else:
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])


        if self.gaussian_filter_flag:

            kernel_sigma = [0.44 * f for f in self.factor]
            kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)

            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x):

        if self.gaussian_filter_flag and self.factor[0] < 1:
            padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
            x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)

        x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class BSplines2DTransform(nn.Module):
    """
    Rescales a transform, which involves resizing the vector field *and* rescaling it.
     Layer for BSplines interpolation with precomputed cubic spline filters.
     It assumes a full sized image from which: (1) it compute the contol points values by downsampling the initial
     image (2) performs the interpolation and (3) crops the image around the valid values.
    :param cp_spacing: _int_ or tuple of three _ints_ specifying the spacing (in pixels) in each dimension.
                      When a single _int_ is used, the same spacing to all dimensions is used
    :param kwargs:
    """

    def __init__(self, vol_shape, cp_spacing, scale=False, **kwargs):
        super().__init__()
        self.cp_spacing = cp_spacing
        self.scale = scale

        b = {
            0: lambda u: np.float((1 - u) ** 3 / 6),
            1: lambda u: np.float((3 * (u ** 3) - 6 * (u ** 2) + 4) / 6),
            2: lambda u: np.float((-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6),
            3: lambda u: np.float(u ** 3 / 6),
        }
        filters = np.zeros(
            (
                2,
                2,
                4 * self.cp_spacing[0],
                4 * self.cp_spacing[1],
            ),
            dtype=np.float,
        )

        for u in range(self.cp_spacing[0]):
            u_norm = 1 - (u + 0.5) / cp_spacing[0]
            for v in range(self.cp_spacing[1]):
                v_norm = 1 - (v + 0.5) / cp_spacing[1]
                for x in range(4):
                    for y in range(4):
                        for it_dim in range(2):
                            filters[
                                it_dim,
                                it_dim,
                                x * self.cp_spacing[0] + u,
                                y * self.cp_spacing[1] + v] = (
                                    b[x](u_norm) * b[y](v_norm)
                            )

        filters = torch.from_numpy(filters).float()

        self.register_buffer('filters', filters)
        self._vol_shape = vol_shape

    def interpolate(self, field):
        """
        :param field: tf.Tensor with shape=number_of_control_points_per_dim
        :return: interpolated_field: tf.Tensor
        """

        # image_shape = tuple([a * b for a, b in zip(field.shape[1:-1], self.cp_spacing)])
        # output_shape = (1,) + image_shape + (1,)
        return F.conv_transpose2d(
            field,
            self.filters,
            bias=None,
            stride=self.cp_spacing
        )

    def forward(self, inputs,  **kwargs):
        """
        :param inputs: tf.Tensor defining a free-form deformation field
        :param kwargs:
        :return: interpolated_field: tf.Tensor of shape=self.input_shape
        """

        vol_shape = self._vol_shape
        high_res_field = self.interpolate(inputs)
        index = [int(3 * c) for c in self.cp_spacing]

        if self.scale:
            for it_dim, factor in enumerate(self.cp_spacing):
                high_res_field[:, it_dim] = factor * high_res_field[:, it_dim]

        return high_res_field[
            :,
            :,
            index[0]: index[0] + vol_shape[0],
            index[1]: index[1] + vol_shape[1],
        ]

class BSplines2DTransformDownsample(nn.Module):
    """
    Rescales a transform, which involves resizing the vector field *and* rescaling it.
     Layer for BSplines interpolation with precomputed cubic spline filters.
     It assumes a full sized image from which: (1) it compute the contol points values by downsampling the initial
     image (2) performs the interpolation and (3) crops the image around the valid values.
    :param cp_spacing: _int_ or tuple of three _ints_ specifying the spacing (in pixels) in each dimension.
                      When a single _int_ is used, the same spacing to all dimensions is used
    :param kwargs:
    """

    def __init__(self, cp_spacing, **kwargs):
        super().__init__()
        self.cp_spacing = cp_spacing

        b = {
            0: lambda u: np.float((1 - u) ** 3 / 6),
            1: lambda u: np.float((3 * (u ** 3) - 6 * (u ** 2) + 4) / 6),
            2: lambda u: np.float((-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6),
            3: lambda u: np.float(u ** 3 / 6),
        }
        filters = np.zeros(
            (
                2,
                2,
                4 * self.cp_spacing[0],
                4 * self.cp_spacing[1],
            ),
            dtype=np.float,
        )

        for u in range(self.cp_spacing[0]):
            u_norm = 1 - (u + 0.5) / cp_spacing[0]
            for v in range(self.cp_spacing[1]):
                v_norm = 1 - (v + 0.5) / cp_spacing[1]
                for x in range(4):
                    for y in range(4):
                        for it_dim in range(2):
                            filters[
                                it_dim,
                                it_dim,
                                x * self.cp_spacing[0] + u,
                                y * self.cp_spacing[1] + v] = (
                                    b[x](u_norm) * b[y](v_norm)
                            )

        filters = torch.from_numpy(filters).float()

        kernel_sigma = [0.44 * cp for cp in self.cp_spacing]
        kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)

        self.register_buffer('filters', filters)
        self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def interpolate(self, field):
        """
        :param field: tf.Tensor with shape=number_of_control_points_per_dim
        :return: interpolated_field: tf.Tensor
        """

        # image_shape = tuple([a * b for a, b in zip(field.shape[1:-1], self.cp_spacing)])
        # output_shape = (1,) + image_shape + (1,)
        return F.conv_transpose2d(
            field,
            self.filters,
            bias=None,
            stride=self.cp_spacing
        )

    def get_control_points(self, field):
        """
        :param field: tf.Tensor of shape=self.input_shape
        :return: interpolated_field: tf.Tensor
        """
        padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
        field = F.conv2d(field, self.kernel, stride=(1, 1), padding=padding)

        vol_shape = field.shape[2:]
        mesh_shape = [
            np.ceil(v / c).astype(int) + 3
            for v, c in zip(vol_shape, self.cp_spacing)
        ]

        return F.interpolate(field, align_corners=True, size=mesh_shape, mode='bilinear')

    def forward(self, inputs, **kwargs):
        """
        :param inputs: tf.Tensor defining a free-form deformation field
        :param kwargs:
        :return: interpolated_field: tf.Tensor of shape=self.input_shape
        """
        vol_shape = inputs.shape[2:]
        low_res_field = self.get_control_points(inputs)
        high_res_field = self.interpolate(low_res_field)
        index = [int(3 * c) for c in self.cp_spacing]
        return high_res_field[
            :,
            :,
            index[0]: index[0] + vol_shape[0],
            index[1]: index[1] + vol_shape[1],
        ]


########################################
##   Normalization layers
########################################
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)






import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from src.layers import  ConvBlock2D, SpatialTransformer, VecInt, ResizeTransform, RescaleTransform


class BaseModel(nn.Module):
    pass

########################################
##   UNet like
########################################

class Unet(BaseModel):
    """
    Voxelmorph Unet. For more information see voxelmorph.net
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, activation='lrelu'):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = self._default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock2D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

    def _default_unet_features(self):
        nb_features = [
            [16, 32, 32, 32],  # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
        return nb_features

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

########################################
##   Dense Registration/Deformation
########################################

class RegNet(nn.Module):
    """
    RegNet network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        use_probs=False,
        gaussian_filter_flag=True,
        rescaling=True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1

        if rescaling:
            self.resize = RescaleTransform(inshape, factor=1 / int_downsize, gaussian_filter_flag=gaussian_filter_flag) if resize else None
            self.fullsize = RescaleTransform(inshape, factor=int_downsize, gaussian_filter_flag=False) if resize else None
        else:
            self.resize = ResizeTransform(inshape, factor=1 / int_downsize, gaussian_filter_flag=gaussian_filter_flag) if resize else None
            self.fullsize = ResizeTransform(inshape, factor=int_downsize, gaussian_filter_flag=False) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        if self.resize:
            flow_field = self.resize(flow_field)

        preint_flow = flow_field

        # integrate to produce diffeomorphic warp
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        # warp image with flow field
        y_source = self.transformer(source, flow_field)

        # return non-integrated flow field if training
        if not registration:
            return y_source, flow_field, preint_flow
        else:
            return y_source, flow_field

    def predict(self, image, flow, diffeomorphic=True, **kwargs):

        if diffeomorphic:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field, nsteps=0):
        if self.integrate:
            flow_field = self.integrate(flow_field, nsteps=nsteps)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field

    def get_composite_field(self, field_1, field_2):
        output = field_1 + self.transformer(field_2, field_1)

        return output


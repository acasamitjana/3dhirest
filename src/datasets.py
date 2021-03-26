# py

# third party imports
import torch
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_dilation
from skimage.exposure import match_histograms
import numpy as np

#project imports
from database.data_loader import EmptySlice
from src.utils import image_transform as tf
from src.utils.image_utils import one_hot_encoding

class RegistrationDataset(Dataset):
    def __init__(self, data_loader, rotation_params, nonlinear_params, tf_params=None, da_params=None, norm_params=None,
                 hist_match=False, mask_dilation=None, to_tensor=True, landmarks=False, num_classes=-1,
                 sbj_per_epoch=1, train=True):
        '''

        :param data_loader:
        :param rotation_params:
        :param nonlinear_params:
        :param tf_params:
        :param da_params:
        :param norm_params:
        :param hist_match:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param num_classes: (int) number of classes for one-hot encoding. If num_classes=-1, one-hot is not performed.
        :param train:
        '''

        self.data_loader = data_loader
        self.subject_list = data_loader.subject_list
        self.N = len(self.subject_list)
        self.to_tensor = to_tensor

        self.landmarks=landmarks

        self.tf_params = tf.Compose(tf_params) if tf_params is not None else None
        self.da_params = tf.Compose_DA(da_params) if da_params is not None else None
        self.norm_params = norm_params if norm_params is not None else lambda x: x
        self.mask_dilation = mask_dilation

        self.hist_match = hist_match
        self.rotation_params = rotation_params
        self.nonlinear_params = nonlinear_params
        self.num_classes = num_classes

        self._image_shape = self.tf_params._compute_data_shape(data_loader.image_shape) if tf_params is not None else data_loader.image_shape
        self.n_dims = data_loader.n_dims
        self.n_channels = data_loader.n_channels
        self.train = train
        self.sbj_per_epoch = sbj_per_epoch


    def mask_image(self, image, mask):
        ndim = len(image.shape)

        if ndim == 3:
            for it_z in range(image.shape[-1]):
                image[..., it_z] = image[..., it_z] * mask
        else:
            image = image*mask

        return image

    def get_ref_data(self, slice, rgb=False, *args, **kwargs):

        x_ref = slice.load_ref(rgb=rgb, *args, **kwargs)
        x_ref_mask = slice.load_ref_mask(*args, **kwargs)
        x_ref_labels = slice.load_ref_labels(*args, **kwargs)
        x_ref_landmarks = slice.load_ref_landmarks() if self.landmarks is True else None
        x_ref = self.mask_image(x_ref,x_ref_mask)


        if np.sum(x_ref_mask) > 0:
            x_ref = self.norm_params(x_ref)
            x_ref = self.mask_image(x_ref, x_ref_mask)

        return x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks

    def get_flo_data(self, slice, rgb=False, *args, **kwargs):

        x_flo = slice.load_flo(rgb=rgb, *args, **kwargs)
        x_flo_mask = slice.load_flo_mask(*args, **kwargs)
        x_flo_labels = slice.load_flo_labels(*args, **kwargs)
        x_flo_landmarks = slice.load_flo_landmarks() if self.landmarks is True else None
        x_flo = self.mask_image(x_flo, x_flo_mask)

        if np.sum(x_flo_mask) > 0:
            x_flo = self.norm_params(x_flo)
            x_flo = self.mask_image(x_flo, x_flo_mask)

        return x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks

    def get_intermodal_data(self, slice, rgb=False, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks = self.get_ref_data(slice, rgb=rgb, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks = self.get_flo_data(slice, rgb=rgb, *args, **kwargs)
        x_flo_orig = x_flo

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0):
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img


        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels,'x_flo_labels': x_flo_labels,
            'x_ref_landmarks':x_ref_landmarks, 'x_flo_landmarks': x_flo_landmarks
        }

        return data_dict

    def get_intramodal_data(self, slice_ref, slice_flo, rgb=False, *args, **kwargs):

        x_ref, x_ref_mask, x_ref_labels, x_ref_landmarks = self.get_ref_data(slice_ref, rgb=rgb, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels, x_flo_landmarks = self.get_flo_data(slice_flo, rgb=rgb, *args, **kwargs)
        x_flo_orig = x_flo

        if self.hist_match and (np.sum(x_ref_mask) > 0 or np.sum(x_flo_mask) > 0):
            x_flo = np.max(x_flo) - x_flo
            x_flo_vec = x_flo[x_flo_mask > 0]
            x_ref_vec = x_ref[x_ref_mask > 0]
            x_flo[x_flo_mask>0] = match_histograms(x_flo_vec, x_ref_vec)

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels,
            'x_ref_landmarks': x_ref_landmarks, 'x_flo_landmarks': x_flo_landmarks

        }

        return data_dict

    def get_deformation_field(self, num=2):
        tf_rot = tf.Rotation(self.rotation_params)
        tf_nonlinear = tf.NonLinearDeformationManyImages(self.nonlinear_params)
        angle_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            angle = tf_rot._get_angle(1)

            nlf_x, nlf_y = tf_nonlinear._get_lowres_strength()
            nonlinear_field = np.zeros((2,) + nlf_x.shape)
            nonlinear_field[0] = nlf_y
            nonlinear_field[1] = nlf_x

            angle_list.append(angle)
            nonlinear_field_list.append(nonlinear_field)

        return angle_list, nonlinear_field_list

    def data_augmentation(self, data_dict):
        x_ref = data_dict['x_ref']
        x_ref_mask = data_dict['x_ref_mask']
        x_ref_labels = data_dict['x_ref_labels']
        x_flo = data_dict['x_flo']
        x_flo_mask = data_dict['x_flo_mask']
        x_flo_labels = data_dict['x_flo_labels']

        if self.da_params is not None:
            img = self.da_params([x_ref, x_ref_mask, x_ref_labels], mask_flag=[False, True, True])
            x_ref, x_ref_mask, x_ref_labels = img
            x_ref = x_ref * x_ref_mask
            x_ref[np.isnan(x_ref)] = 0
            x_ref_mask[np.isnan(x_ref_mask)] = 0
            x_ref_labels[np.isnan(x_ref_labels)] = 0

            img = self.da_params([x_flo, x_flo_mask, x_flo_labels], mask_flag=[False, True, True])
            x_flo, x_flo_mask, x_flo_labels = img
            x_flo = x_flo * x_flo_mask
            x_flo[np.isnan(x_flo)] = 0
            x_flo_mask[np.isnan(x_flo_mask)] = 0
            x_flo_labels[np.isnan(x_flo_labels)] = 0

        if self.mask_dilation is not None:
            x_ref_mask = binary_dilation(x_ref_mask, structure=self.mask_dilation)
            x_flo_mask = binary_dilation(x_flo_mask, structure=self.mask_dilation)

        data_dict['x_ref'] = x_ref
        data_dict['x_ref_mask'] = x_ref_mask
        data_dict['x_ref_labels'] = x_ref_labels
        data_dict['x_flo'] = x_flo
        data_dict['x_flo_mask'] = x_flo_mask
        data_dict['x_flo_labels'] = x_flo_labels

        return data_dict

    def convert_to_tensor(self, data_dict):

        if not self.to_tensor:
            return data_dict

        for k, v in data_dict.items():
            if 'landmarks' in k:
                continue
            elif 'labels' in k and self.num_classes:
                v = one_hot_encoding(v, self.num_classes) if self.num_classes else v
                data_dict[k] = torch.from_numpy(v).float()
            elif isinstance(v, list):
                data_dict[k] = [torch.from_numpy(vl).float() for vl in v]
            else:
                data_dict[k] = torch.from_numpy(v[np.newaxis]).float()

        return data_dict

    def __len__(self):
        return self.sbj_per_epoch * self.N

class InterModalRegistrationDataset(RegistrationDataset):
    '''
    Class for intermodal registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, rotation_params, nonlinear_params, tf_params=None,
                 da_params=None, norm_params=None, hist_match=False, mask_dilation=None, landmarks=False, train=True,
                 num_classes=False, to_tensor=True, sbj_per_epoch=1):

        super().__init__(data_loader, rotation_params, nonlinear_params, tf_params=tf_params, da_params=da_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.hist_match = hist_match

    def __getitem__(self, index):

        subject = self.subject_list[index]

        if hasattr(subject, 'get_slice'):
            slice, slice_num = subject.get_slice(random_seed=None if self.train else 44)
            rid = slice.id
        else:
            slice = subject
            rid = subject.id

        data_dict = self.get_intermodal_data(slice)
        data_dict = self.data_augmentation(data_dict)

        angle, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)

        x_ref = data_dict['x_ref']
        x_ref_mask = data_dict['x_ref_mask']
        x_ref_labels = data_dict['x_ref_labels']
        x_flo = data_dict['x_flo']
        x_flo_mask = data_dict['x_flo_mask']
        x_flo_labels = data_dict['x_flo_labels']
        nonlinear_field = data_dict['nonlinear']

        return x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels, angle, nonlinear_field, rid

    @property
    def image_shape(self):
        return self._image_shape

class IntraModalRegistrationDataset(RegistrationDataset):
    '''
    Basic class for registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''

    def __init__(self, data_loader, rotation_params, nonlinear_params, tf_params=None, da_params=None,
                 norm_params=None, hist_match=False, mask_dilation=None, to_tensor=True, landmarks=False, train=True,
                 num_classes=False, sbj_per_epoch=1, neighbor_distance=-1, fix_neighbors=False):

        '''
        :param data_loader:
        :param transform_parameters:
        :param data_augmentation_parameters:
        :param normalization:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param train:
        :param neighbor_distance: forward distance in mm (positive) or in number of neighbors (negative)
        '''
        super().__init__(data_loader, rotation_params, nonlinear_params, tf_params=tf_params, da_params=da_params,
                         norm_params=norm_params, train=train, hist_match=hist_match, mask_dilation=mask_dilation,
                         to_tensor=to_tensor, num_classes=num_classes, sbj_per_epoch=sbj_per_epoch, landmarks=landmarks)

        self.neighbor_distance = neighbor_distance
        self.fix_neighbors = fix_neighbors

    def get_slice(self, subject_ref, index_ref):

        if hasattr(subject_ref, 'get_slice'):
            slice_ref, index_ref = subject_ref.get_slice(random_seed=None if self.train else 44)
            rid_ref = slice_ref.id
            slice_list = subject_ref.slice_list

        else:
            slice_ref = subject_ref
            rid_ref = subject_ref.id
            slice_list = self.subject_list

        if self.neighbor_distance == 0:
            slice_flo = slice_ref
            rid_flo = rid_ref
            index_flo = index_ref

            return slice_ref, slice_flo, rid_ref, rid_flo, index_flo

        if not self.fix_neighbors: # here we get randomly any neighbor
            if self.neighbor_distance > 0:
                slices_available = [sref for sref in slice_list if
                                    sref.id_mm <= slice_ref.id_mm + self.neighbor_distance and
                                    sref.id_mm >= slice_ref.id_mm - self.neighbor_distance and
                                    sref.id != slice_ref.id]

                index_flo = int(np.random.choice(len(slices_available), size=1))
                slice_flo = slices_available[index_flo]
                rid_flo = slice_flo.id

            else:
                slices_available = slice_list[index_ref+1:index_ref - self.neighbor_distance]
                index_flo = int(np.random.choice(len(slices_available), size=1))
                slice_flo = slices_available[index_flo]
                rid_flo = slice_flo.id

        else:
            if self.neighbor_distance > 0:
                index_mm = slice_ref.id_mm + self.neighbor_distance
                index_flo = [it_sref for it_sref, sref in enumerate(slice_list) if sref.id_mm == index_mm]
                if index_flo:
                    index_flo = index_flo[0]
            else:
                index_flo = index_ref - self.neighbor_distance
                if index_flo >= len(slice_list):  # if it reached the last position, register the other way around.
                    index_flo = []

            if not index_flo:
                slice_flo = EmptySlice(self.image_shape)
            else:
                slice_flo = slice_list[index_flo]

            rid_ref = slice_ref.id
            rid_flo = slice_flo.id

        return slice_ref, slice_flo, rid_ref, rid_flo, index_flo

    def __getitem__(self, index):
        subject_ref = self.subject_list[index]
        slice_ref, slice_flo, rid_ref, rid_flo, _ = self.get_slice(subject_ref, index)
        rid = rid_flo + '_to_' + rid_ref

        data_dict = self.get_intramodal_data(slice_ref, slice_flo)
        data_dict = self.data_augmentation(data_dict)
        angle, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)

        x_ref = data_dict['x_ref']
        x_ref_mask = data_dict['x_ref_mask']
        x_ref_labels = data_dict['x_ref_labels']
        x_flo = data_dict['x_flo']
        x_flo_mask = data_dict['x_flo_mask']
        x_flo_labels = data_dict['x_flo_labels']
        nonlinear_field = data_dict['nonlinear']

        return x_ref, x_flo, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels, angle, nonlinear_field, rid

    @property
    def image_shape(self):
        return self._image_shape

    def __len__(self):
        if self.neighbor_distance < 0:
            return super().__len__() + self.neighbor_distance
        else:
            return super().__len__()


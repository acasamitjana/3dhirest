import csv
from os.path import join
import nibabel as nib

import numpy as np
from PIL import Image


class EmptySlice(object):

    def __init__(self, shape):

        self.shape = shape

    def load_ref(self, *args, **kwargs):
        return np.zeros(self.shape)

    def load_flo(self, *args, **kwargs):
        return np.zeros(self.shape)

    def load_ref_mask(self,*args, **kwargs):
        return np.zeros(self.shape)

    def load_flo_mask(self, *args, **kwargs):
        return np.zeros(self.shape)

    def load_ref_labels(self, *args, **kwargs):
        return np.zeros(self.shape)

    def load_flo_labels(self, *args, **kwargs):
        return np.zeros(self.shape)

    @property
    def id(self):
        return 0

    @property
    def id_mm(self):
        return 0

    @property
    def image_shape(self):
        return self.shape

class Slice(object):

    def __init__(self, subject, slice_name, slice_num, tree_pos, **kwargs):

        self.subject = subject
        self._sid = str(subject.id) + '.s' + slice_num
        self.tree_pos = int(tree_pos)

        self.slice_name = slice_name
        self.slice_num = int(slice_num)

        self._image_shape = None


    @property
    def slice_name_ref(self):
        if self.subject.ref_modaltiy == 'IHC':
            return self.slice_name
        else:
            return 'slice_' + "{:03d}".format(self.tree_pos+1) + '.png'

    @property
    def slice_name_flo(self):
        if self.subject.flo_modality == 'IHC':
            return self.slice_name
        else:
            return 'slice_' + "{:03d}".format(self.tree_pos + 1) + '.png'


    def load_ref(self, *args, **kwargs):

        data = Image.open(join(self.subject.dir_ref_image, self.slice_name_ref))
        data = np.array(data)
        return data

    def load_flo(self, *args, **kwargs):

        data = Image.open(join(self.subject.dir_flo_image, self.slice_name_flo))
        data = np.array(data)
        return data


    def load_ref_mask(self,*args, **kwargs):
        data = Image.open(join(self.subject.dir_ref_mask, self.slice_name_ref))
        data = np.double(np.array(data)>0)

        return data

    def load_flo_mask(self, *args, **kwargs):
        data = Image.open(join(self.subject.dir_flo_mask, self.slice_name_flo))
        data = np.double(np.array(data) > 0)

        return data

    def load_ref_labels(self, *args, **kwargs):
        return self.load_ref_mask(*args, **kwargs)

    def load_flo_labels(self, *args, **kwargs):

        return self.load_flo_mask(*args, **kwargs)

    @property
    def id(self):
        return self._sid

    @property
    def id_mm(self):
        return float(self.tree_pos)

    @property
    def image_shape(self):
        if self._image_shape is None:
            self._image_shape = self.load_ref().shape

        return self._image_shape


class Subject(object):
    def __init__(self, sbj_id, data_path, dataset_sheet, ref_modality, flo_modality):

        self.data_path = data_path
        self.dataset_sheet = dataset_sheet
        self._id = sbj_id
        self._slice_list = []

        self.dir_ref_image = join(data_path, ref_modality.lower(), 'images')
        self.dir_ref_mask = join(data_path, ref_modality.lower(), 'masks')
        self.dir_flo_image = join(data_path, flo_modality.lower(), 'images')
        self.dir_flo_mask = join(data_path, flo_modality.lower(), 'masks')

        self.ref_modaltiy = ref_modality
        self.flo_modality = flo_modality
        self._slice_list = []

    def initialize_subject(self):
        slice_list = []
        with open(self.dataset_sheet, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                slice_list.append(Slice(self, row['filename'], row['slice_number'], tree_pos=row['tree_pos']))

        self._slice_list = slice_list

    def add_slice(self, slice_name, slice_num, tree_pos, slice_id_mm=None):
        if slice_id_mm is None:
            slice_i = Slice(self, slice_name, slice_num, tree_pos)

        else:
            slice_i = Slice(self, slice_name, slice_num, tree_pos, slice_id_mm=slice_id_mm)

        self.slice_list.append(slice_i)

    def get_slice(self, slice_num=None, random_seed=44):

        if slice_num is None:
            np.random.seed(random_seed)
            slice_num = np.random.choice(len(self._slice_list), 1, replace=False)[0]

        s = self._slice_list[slice_num]

        return s

    def load_mri_volume(self):
        proxy = nib.load(join(self.data_path, 'linear', 'mri.nii.gz'))
        data = np.asarray(proxy.dataobj)

        return data

    def load_mri_mask_volume(self):
        proxy = nib.load(join(self.data_path, 'linear', 'mri.nii.gz'))
        data = np.asarray(proxy.dataobj) > 0

        return data

    def __len__(self):
        return len(self._slice_list)

    @property
    def slice_list(self):
        return self._slice_list

    @property
    def image_shape(self):
        return self._slice_list[0].image_shape

    @property
    def id(self):
        return self._id

    @property
    def mri_affine(self):
        proxy = nib.load(join(self.data_path, 'linear', 'mri.nii.gz'))
        return proxy.affine

    @property
    def pre_empty_slices(self):
        return 25

    @property
    def post_empty_slices(self):
        return 25

    @property
    def num_tree_pos(self):
        return max([s.tree_pos for s in self._slice_list]) + 1

class DataLoader(object):

    def __init__(self, parameter_dict, **kwargs):

        self.database_config = parameter_dict['DB_CONFIG']
        self.ref_modality = parameter_dict['REF_MODALITY']
        self.flo_modality = parameter_dict['FLO_MODALITY']

        self.data_path = self.database_config['BASE_DIR']
        self.dataset_sheet = self.database_config['DATASET_SHEET']

        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, **kwargs):
        self.n_dims = 2
        self.n_channels = 1

        subject = Subject('0001', self.data_path, self.dataset_sheet, self.ref_modality, self.flo_modality)
        subject.initialize_subject()

        self.subject_dict = {'0001': subject}
        self.rid_list = ['0001']
        self.subject_list = [self.subject_dict['0001']]

    @property
    def image_shape(self):
        return self.subject_list[0].image_shape

    def __len__(self):
        return len(self.rid_list)

class DataLoaderBlock(object):

    def __init__(self, parameter_dict, subject_id_key='block_id', **kwargs):
        self.database_config = parameter_dict['DB_CONFIG']
        self.ref_modality = parameter_dict['REF_MODALITY']
        self.flo_modality = parameter_dict['FLO_MODALITY']

        # self.subject_id_key = subject_id_key
        self.data_path = self.database_config['BASE_DIR']
        self._initialize_dataset(subject_id_key, **kwargs)

    def _initialize_dataset(self, subject_id_key, bid_list=None):
        self.n_dims = 2
        self.n_channels = 1
        self.subject_dict = {}

        with open(self.database_config['DATASET_SHEET'], 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if bid_list is not None:
                    if row[subject_id_key] not in bid_list:
                        continue

                if row[subject_id_key] not in self.subject_dict.keys():
                    self.subject_dict[row[subject_id_key]] = Subject(row[subject_id_key],
                                                                     self.data_path,
                                                                     self.database_config['DATASET_SHEET'],
                                                                     self.ref_modality, self.flo_modality)

                self.subject_dict[row[subject_id_key]].add_slice(
                    slice_name=row['filename'],
                    slice_num=row['slice_number'],
                    tree_pos=row['tree_pos'],
                )

        self.rid_list = [sbj for sbj in self.subject_dict.keys()]
        self.subject_list = [sbj for sbj in self.subject_dict.values()]


    @property
    def image_shape(self):
        return self.subject_list[0].image_shape

    def __len__(self):
        return len(self.rid_list)

from os.path import join, dirname, basename

import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression

from scripts import configFile
from src.utils.image_utils import build_regular_space_volume, build_regular_space_volume_color


NSLICES = configFile.NSLICES
MRI_DIR = '/Users/acasamitjana/Data/Allen_data/'
FILE_LIST = [
    '/Users/acasamitjana/Data/Allen_data/ihc//IHC_IMAGE_0.25.nii.gz'
]
input_z_pos = '/home/acasamitjana/Results/Registration/Allen/Rigid_def/Second_it/z_pos_edges_lncc.lin.npy'
z_pos = 1 * np.squeeze(np.load(input_z_pos))

outliers = (140, 170)
target_sp = 0.25
image_shape = (448, 320)
affine_H = np.eye(4)
brightness_f = lambda R,G,B: 0.2126 * R + 0.7152 * G + 0.0722 * B
colored=False
for stain in FILE_LIST:
    print(stain)

    median_H = []
    median_M = []

    proxy = nib.load(stain)
    affine_H = proxy.affine
    image_shape = proxy.shape[:2]
    data_H = np.asarray(proxy.dataobj)

    proxy = nib.load(join(MRI_DIR, 'MRI_images.nii.gz'))
    data_M = np.asarray(proxy.dataobj)

    if data_H.shape[-1] == 3:
        colored = True

    for it_s in range(NSLICES):
        H = brightness_f(data_H[..., it_s, 0], data_H[..., it_s, 1], data_H[..., it_s, 2]) if colored else data_H[..., it_s]
        M = data_M[..., it_s]

        H_M = H > 0
        M_M = M > 0
        m = np.median(M[M_M])
        if m > outliers[0] and m < outliers[1] and np.sum(H_M) > 0:
            h = np.median(H[H_M])
            median_H.append(h)
            median_M.append(m)

    lr = LinearRegression()
    lr.fit(np.asarray(median_M).reshape(-1,1), np.asarray(median_H).reshape(-1,1))
    a = lr.coef_
    b = lr.intercept_

    data_H_eq = np.zeros(image_shape + (NSLICES,3)) if colored else np.zeros(image_shape + (NSLICES,))
    for it_s in range(NSLICES):
        H = brightness_f(data_H[..., it_s, 0], data_H[..., it_s, 1], data_H[..., it_s, 2]) if colored else data_H[..., it_s]
        M = data_M[..., it_s]

        H_M = H > 0
        M_M = M > 0
        m = np.median(M[M_M])
        if m > 140 and m < 170  and np.sum(H_M) > 0:
            h = np.median(H[H_M])
            new_h = a * m + b
            if colored:
                new_H = new_h / h * data_H[..., it_s, :]
                data_H_eq[..., it_s,:] = np.clip(new_H, 0, 255)
            else:
                new_H = new_h/h*H
                data_H_eq[..., it_s] = np.clip(new_H, 0, 255)

        else:
            print(it_s)

    Z_RES = 0.25

    INIT_RES = configFile.INIT_RES
    BLOCK_RES = configFile.BLOCK_RES
    MRI_AFFINE = configFile.MRI_AFFINE
    CROP_XY = configFile.CROP_XY
    Trot = np.array([[0, 0, -1, -CROP_XY[0]],
                     [-1, 0, 0, -CROP_XY[1]],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]])  # brings MRI into the orientation of the stack
    TrotInv = np.linalg.inv(Trot)

    inplane_factor = INIT_RES / BLOCK_RES
    aux = np.asarray([(inplane_factor - 1) / (2 * inplane_factor)] * 2 + [0])
    cog_xy = MRI_AFFINE[:3, 3] - np.dot(MRI_AFFINE[:3, :3], aux.T)


    if colored:
        vol, aff = build_regular_space_volume_color(data_H_eq, z_pos, cog_xy, BLOCK_RES, target_sp=Z_RES)
        vol = (vol-150) * 255/105
        vol = np.clip(vol, 0, 255)
    else:
        vol, aff = build_regular_space_volume(data_H_eq, z_pos, cog_xy, BLOCK_RES, target_sp=Z_RES)

    img = nib.Nifti1Image(vol, np.matmul(TrotInv, aff))
    nib.save(img, join(dirname(stain), basename(stain) + '.eq_easy.resampled.' + str(Z_RES) + '.nii.gz'))

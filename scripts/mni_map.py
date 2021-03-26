import subprocess
from os.path import join, exists
from os import makedirs

import nibabel as nib
import pdb
import numpy as np
from scipy.interpolate import interpn

from setup import DATA_DIR, RESULTS_DIR, NIFTY_REG_DIR



###########
## Files ##
###########

# T2 image from Allen subject and MNI template
MRI = join(DATA_DIR, 'downloads', 'linear', 'mri.nii.gz')
MNI_TEMPLATE = join(DATA_DIR, 'downloads', 'mni_icbm152_nlin_sym_09b', 'mni_icbm152_t2_tal_nlin_sym_09b_hires.nii')
MNI_ASEG_TEMPLATE = join(DATA_DIR, 'downloads', 'mni_icbm152_nlin_sym_09b', 't1_fs', 'mri', 'aseg.mgz')

# Left hemisphere of the MNI template
lh_dir= join(DATA_DIR, 'dataset', 'mni_lh')
if not exists(lh_dir):
    makedirs(lh_dir)

MNI_LH = join(lh_dir, 't2.nii.gz')
MNI_LH_ALIGNED = join(lh_dir, 't2.aligned.nii.gz')
MNI_LH_MASK = join(lh_dir, 't2.mask.nii.gz')
MNI_LH_ASEG = join(lh_dir, 't2.aseg.nii.gz')


# MNI registration files
results_dir = join(RESULTS_DIR, 'mni')
if not exists(results_dir):
    makedirs(results_dir)
reg_volume_aladin = join(results_dir, 'mni.t2.registered.aladin.nii.gz')
reg_volume = join(results_dir, 'mni.t2.registered.nii.gz')
affine_file = join(results_dir, 'aladin.affine.txt')
nonLinearDisplacementField = join(results_dir, 'mni.displacement.nii.gz')
affineDisplacementField = join(results_dir, 'mni.affine.nii.gz')
totalDeformationField = join(results_dir, 'mni.coords.ras.nii.gz')
totalDeformationField_IJK = join(results_dir, 'mni.coords.ijk.nii.gz')


############
## STEP 1 ##
############
# Run freesurfer with the appropriate file paths:
# export SUBJECTS_DIR=join(results_dir, )
# recon-all -subjid t1_fs -i join(DATA_DIR, 'downloads', 'mni_icbm152_nlin_sym_09b',
#                                 'mni_icbm152_t2_tal_nlin_sym_09b_hires.nii') -all


############
## STEP 2 ##
############
print("Computing the left hemisphere from the MNI template")
mni_proxy = nib.load(MNI_TEMPLATE)
mni = np.asarray(mni_proxy.dataobj) / 255. / 0.4
mni_affine = mni_proxy.affine
mni_shape = mni_proxy.shape

mni_aseg_proxy = nib.load(MNI_ASEG_TEMPLATE)
mni_aseg = np.asarray(mni_aseg_proxy.dataobj)
fs_affine = mni_aseg_proxy.affine
fs_shape = mni_aseg_proxy.shape

mni_lh_mask = np.ones_like(mni_aseg)
mni_lh_mask[np.where(mni_aseg == 0)] = 0
mni_lh_mask[np.where(mni_aseg > 40)] = 0
mni_lh_mask[np.where(mni_aseg == 77)] = 1
mni_lh_mask[:127] = 0

II, JJ, KK = np.meshgrid(np.arange(0, mni_shape[0]), np.arange(0, mni_shape[1]), np.arange(0, mni_shape[2]), indexing='ij')
voxMosaic = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((np.prod(mni_shape), 1))), axis=1).T
voxB = np.dot(np.linalg.inv(fs_affine), np.dot(mni_affine, voxMosaic))
vR = voxB[0]
vA = voxB[1]
vS = voxB[2]
ok1 = vR >= 0
ok2 = vA >= 0
ok3 = vS >= 0
ok4 = vR <= fs_shape[0] - 1
ok5 = vA <= fs_shape[1] - 1
ok6 = vS <= fs_shape[2] - 1
ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

xiR = np.reshape(vR[ok], (-1, 1))
xiA = np.reshape(vA[ok], (-1, 1))
xiS = np.reshape(vS[ok], (-1, 1))
xi = np.concatenate((xiR, xiA, xiS), axis=1)
mni_lh_mask_resampled_f = np.zeros(vR.shape[0])
mni_lh_mask_resampled_f[ok] = interpn((np.arange(0, fs_shape[0]),
                                 np.arange(0, fs_shape[1]),
                                 np.arange(0, fs_shape[2])), mni_lh_mask, xi=xi)

mni_lh_mask_resampled = mni_lh_mask_resampled_f.reshape(mni_shape)
mni_lh = np.zeros_like(mni)
mni_lh[mni_lh_mask_resampled>0] = mni[mni_lh_mask_resampled>0]

mni_lh_aseg = np.zeros_like(mni_aseg)
mni_lh_aseg[mni_lh_mask>0] = mni_aseg[mni_lh_mask>0]

img = nib.Nifti1Image(mni_lh, mni_affine)
nib.save(img, MNI_LH)
img = nib.Nifti1Image(mni_lh_mask, fs_affine)
nib.save(img, MNI_LH_MASK)
img = nib.Nifti1Image(mni_lh_aseg, fs_affine)
nib.save(img, MNI_LH_ASEG)

############
## STEP 3 ##
############

ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps/reg_transform'
RESAMPLEcmd = NIFTY_REG_DIR + 'reg-apps/reg_resample'
dummyFileNifti1 = '/tmp/dummyfileNifty1.nii.gz'
dummyFileNifti2 = '/tmp/dummyfileNifty2.nii.gz'


print("Computing initial alignment: reg_aladin")
subprocess.call([ALADINcmd, '-ref', MRI, '-flo', MNI_LH, '-aff', affine_file, '-res', reg_volume_aladin,
                 '-ln', '4', '-lp', '3', '-pad', '0', '-speeeeed'], stdout=subprocess.DEVNULL)




print("Computing non-linear registration. reg_f3d")
scontrol = [-8, -8, -8]
subprocess.call(
    [F3Dcmd, '-ref', MRI, '-flo', reg_volume_aladin, '-res', reg_volume, '-cpp', dummyFileNifti1,
     '-sx', str(scontrol[0]), '-sy', str(scontrol[1]), '-sz', str(scontrol[2]), '-ln', '4', '-lp', '3',
     '--nmi', '-be', str(0.01), '-pad', '0', '-vel'], stdout=subprocess.DEVNULL)



print("Computing the resulting deformation field")
subprocess.call([TRANSFORMcmd, '-ref', MRI, '-disp', affine_file, affineDisplacementField], stdout=subprocess.DEVNULL)
subprocess.call([TRANSFORMcmd, '-ref', MRI, '-disp', dummyFileNifti1, nonLinearDisplacementField], stdout=subprocess.DEVNULL)
subprocess.call([TRANSFORMcmd, '-ref', MRI, '-ref2', MRI, '-comp', dummyFileNifti1, affine_file, totalDeformationField], stdout=subprocess.DEVNULL)



print("Transforming MNI coordinates from RAS to XYZ")
proxy = nib.load(totalDeformationField)
total_def = np.asarray(proxy.dataobj)
affine = proxy.affine

proxy = nib.load(MNI_TEMPLATE)
vox2ras_mni = proxy.affine
vox2ras_mni_inv = np.linalg.inv(vox2ras_mni)

total_def_T = np.transpose(total_def, [4,0,1,2,3])
ones = np.ones((1,) + total_def.shape[:-1])
ras1 = np.concatenate((total_def_T, ones), axis=0)
ras1_flat = ras1.reshape((4,-1))

xyz1_flat = np.matmul(vox2ras_mni_inv, ras1_flat)
xyz1 = xyz1_flat.reshape(ras1.shape)
xyz = xyz1[:3]

total_def_xyz = np.transpose(xyz,[1,2,3,4,0])
img = nib.Nifti1Image(total_def_xyz, affine)
nib.save(img, totalDeformationField_IJK)
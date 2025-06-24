import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import nibabel as nib
import os
import numpy as np
import numpy as np
import scipy as sp
import porespy as ps
import SimpleITK as sitk
from skimage.measure import regionprops
import random
import cv2
import shutil
from scipy.ndimage import binary_fill_holes, gaussian_filter, label, zoom
from torch.nn.functional import threshold
from triton.language import dtype
from bulls_eye_plot import *

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
data_id = 'N5'
sub_folder = './Dataset/emidec_dataset/train_original/'
in_folder = './GenMasks1/'+data_id+'/emidec_dataset/train/'
out_folder = './GenMasks_Subject_Space1/'+data_id+'/emidec_dataset/train/'
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def main():
    # get all data
    input_masks, sub_N_masks, sub_N_images, sub_P_masks, sub_P_images = get_data(in_folder, sub_folder)
    # --------------------------------------------------------------------------------
    # apply rigid registration
    reg1_masks = apply_rigid_registration(input_masks, sub_N_masks, sub_N_images, out_folder)
    # --------------------------------------------------------------------------------
    # apply non-rigid registration
    reg2_masks = apply_non_rigid_registration(reg1_masks, sub_N_masks, sub_N_images, out_folder)
    # --------------------------------------------------------------------------------
    # correct the generated scar masks (limit within the original myocardium)
    correct_scar_masks(reg2_masks, sub_N_masks, sub_N_images, out_folder)

#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

def get_data(in_folder, sub_folder):
    input_files = []
    for folder_path in os.listdir(in_folder):
        if ('Case' in folder_path):
            mask_file_path = os.path.join(in_folder, folder_path + '/Contours')
            input_files.append(get_all_images(mask_file_path, 'gz'))
        else:
            continue
    input_masks = []
    for i in range(len(input_files)):
        if ('Case_N' in input_files[i][0]):
            gd = [x for x in input_files[i] if ('_scar' not in x)]
        elif ('Case_P' in input_files[i][0]):
            gd = [x for x in input_files[i] if ('_scar' not in x and '_comb' not in x and '_orig' not in x)]
        input_masks.append(gd[0])
    input_masks.sort()
    # --------------------------------------------------------------------------------
    sub_N_files1 = []
    sub_N_files2 = []
    sub_P_files1 = []
    sub_P_files2 =  []
    for folder_path in os.listdir(sub_folder):
        if ('Case_N' in folder_path):
            mask_file_path = os.path.join(sub_folder, folder_path + '/Contours')
            image_file_path = os.path.join(sub_folder, folder_path + '/Images')
            sub_N_files1.append(get_all_images(mask_file_path, 'gz'))
            sub_N_files2.append(get_all_images(image_file_path, 'gz'))
    for folder_path in os.listdir(sub_folder):
        if ('Case_P' in folder_path):
            mask_file_path = os.path.join(sub_folder, folder_path + '/Contours')
            image_file_path = os.path.join(sub_folder, folder_path + '/Images')
            sub_P_files1.append(get_all_images(mask_file_path, 'gz'))
            sub_P_files2.append(get_all_images(image_file_path, 'gz'))
    sub_N_masks = []
    sub_N_images = []
    sub_P_masks = []
    sub_P_images = []
    for i in range(len(sub_N_files1)):
        gd = sub_N_files1[i]
        img = sub_N_files2[i]
        sub_N_masks.append(gd[0])
        sub_N_images.append(img[0])
    for i in range(len(sub_P_files1)):
        gd = sub_P_files1[i]
        img = sub_P_files2[i]
        sub_P_masks.append(gd[0])
        sub_P_images.append(img[0])
    sub_N_masks.sort()
    sub_N_images.sort()
    sub_P_masks.sort()
    sub_P_images.sort()
    # --------------------------------------------------------------------------------
    return input_masks, sub_N_masks, sub_N_images, sub_P_masks, sub_P_images

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def command_iteration(method):
    """ Callback invoked when the optimization has an iteration """
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
    )

def rigid_registration(fixed, moving):
    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension()))
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    R.SetOptimizerAsGradientDescent(
        learningRate=10,
        numberOfIterations=300,
        estimateLearningRate=R.EachIteration,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initialTx)
    R.SetInterpolator(sitk.sitkNearestNeighbor)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx1 = R.Execute(fixed, moving)
    displacementField = sitk.TransformToDisplacementFieldFilter()
    displacementField.SetReferenceImage(fixed)
    displacementTx = sitk.DisplacementFieldTransform(displacementField.Execute(sitk.Transform(3, sitk.sitkIdentity)))
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
    R.SetMovingInitialTransform(outTx1)
    R.SetInitialTransform(displacementTx, inPlace=True)
    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.MetricUseFixedImageGradientFilterOff()
    R.Execute(fixed, moving)
    print("------------------------------------------")
    print(displacementTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    compositeTx = sitk.CompositeTransform([outTx1, displacementTx])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(compositeTx)
    out = resampler.Execute(moving)
    return out

def apply_rigid_registration(input_masks, sub_masks, sub_images, out_folder):
    reg1_masks = []
    for i in range(len(input_masks)):
        moving_mask_file = input_masks[i]
        fixed_mask_file = sub_masks[i]
        moving_mask = sitk.ReadImage(moving_mask_file, sitk.sitkFloat32)
        fixed_mask = sitk.ReadImage(fixed_mask_file, sitk.sitkFloat32)
        out_mask = rigid_registration(fixed_mask, moving_mask)
        new_name = moving_mask_file.split('/')[-1].split('.')[0]
        folder_path1 = out_folder + new_name + "/Contours/"
        folder_path2 = out_folder + new_name + "/Images/"
        if not os.path.exists(folder_path1):
            os.makedirs(folder_path1)
        if not os.path.exists(folder_path2):
            os.makedirs(folder_path2)
        filename1 = folder_path1 + new_name + "_reg1.nii.gz"
        filename2 = folder_path2 + new_name + "_reg1.nii.gz"
        sitk.WriteImage(out_mask, filename1)
        shutil.copy(sub_images[i], filename2)
        print('--------------Successful--------------------')
        print(i, filename1, filename2)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        reg1_masks.append(filename1)
    return reg1_masks

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def command_iteration1(filter):
    global metric_values
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    metric_values.append(filter.GetMetric())

metric_values = []
def non_rigid_registraion(fixed_image, moving_image):
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(5000)
    demons.SetStandardDeviations(1.2)
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration1(demons))
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                     (sitk.Transform(3, sitk.sitkIdentity))
                                                     )
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
    displacementField = transform_to_displacment_field_filter.Execute(displacementTx)
    displacementField = demons.Execute(fixed_image, moving_image, displacementField)
    outTx = sitk.DisplacementFieldTransform(displacementField)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving_image)
    return out

def apply_non_rigid_registration(input_masks, sub_masks, sub_images, out_folder):
    reg2_masks = []
    for i in range(len(input_masks)):
        moving_mask_file = input_masks[i]
        fixed_mask_file = sub_masks[i]
        moving_mask = sitk.ReadImage(moving_mask_file, sitk.sitkFloat32)
        fixed_mask = sitk.ReadImage(fixed_mask_file, sitk.sitkFloat32)
        out_mask = rigid_registration(fixed_mask, moving_mask)
        new_name = moving_mask_file.split('/')[-1].split('.')[0]
        new_name = new_name.replace('_reg1', '')
        folder_path1 = out_folder + new_name + "/Contours/"
        folder_path2 = out_folder + new_name + "/Images/"
        if not os.path.exists(folder_path1):
            os.makedirs(folder_path1)
        if not os.path.exists(folder_path2):
            os.makedirs(folder_path2)
        filename1 = folder_path1 + new_name + "_reg2.nii.gz"
        filename2 = folder_path2 + new_name + "_reg2.nii.gz"
        sitk.WriteImage(out_mask, filename1)
        shutil.copy(sub_images[i], filename2)
        print('--------------Successful--------------------')
        print(i, filename1, filename2)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        reg2_masks.append(filename1)
    return reg2_masks

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def correct_scar_masks(input_masks, sub_masks, sub_images, out_folder):
    for i in range(len(input_masks)):
        input_mask = nib.load(input_masks[i]).get_fdata()
        sub_mask  = nib.load(sub_masks[i]).get_fdata()
        new_name = input_masks[i].split('/')[-1].split('.')[0]
        new_name = new_name.replace('_reg2', '')
        myo_mask = np.zeros((sub_mask.shape[0], sub_mask.shape[1], sub_mask.shape[2]))
        scar_mask = np.zeros((sub_mask.shape[0], sub_mask.shape[1], sub_mask.shape[2]))
        myo_mask[sub_mask==1] = 0
        myo_mask[sub_mask==2] = 1
        myo_mask[sub_mask==3] = 1
        myo_mask[sub_mask==4] = 1
        scar_mask[input_mask==3] = 1
        scar_mask = scar_mask * myo_mask
        out_mask = sub_mask.copy()
        out_mask[sub_mask==3] = 2
        out_mask[sub_mask==4] = 2
        out_mask[scar_mask==1] = 3
        if ('Case_P' in new_name):
            orig_mask = sub_mask.copy()
            orig_mask[sub_mask==4] = 2
            comb_mask = sub_mask.copy()
            comb_mask[sub_mask==4] = 2
            comb_mask[scar_mask==1] = 3
        folder_path1 = out_folder + new_name + "/Contours/"
        folder_path2 = out_folder + new_name + "/Images/"
        if not os.path.exists(folder_path1):
            os.makedirs(folder_path1)
        if not os.path.exists(folder_path2):
            os.makedirs(folder_path2)
        filename1 = folder_path2 + new_name + ".nii.gz"
        filename2 = folder_path1 + new_name + ".nii.gz"
        filename3 = folder_path1 + new_name + "_scar.nii.gz"
        shutil.copy(sub_images[i], filename1)
        mask1 = nib.Nifti1Image(out_mask, affine=nib.load(sub_images[i]).affine, header=nib.load(sub_images[i]).header)
        nib.save(mask1, filename2)
        mask2 = nib.Nifti1Image(scar_mask, affine=nib.load(sub_images[i]).affine, header=nib.load(sub_images[i]).header)
        nib.save(mask2, filename3)
        if ('Case_P' in new_name):
            filename4 = folder_path1 + new_name + "_comb.nii.gz"
            mask3 = nib.Nifti1Image(comb_mask, affine=nib.load(sub_images[i]).affine, header=nib.load(sub_images[i]).header)
            nib.save(mask3, filename4)
            filename5 = folder_path1 + new_name + "_orig.nii.gz"
            mask4 = nib.Nifti1Image(orig_mask, affine=nib.load(sub_images[i]).affine,header=nib.load(sub_images[i]).header)
            nib.save(mask4, filename5)
        print('--------------Successful--------------------')
        print(i, filename1, filename2)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

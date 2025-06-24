import SimpleITK as sitk
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
train_folder = './Output/emidec_dataset/train/'
test_folder = './Output/emidec_dataset/test/'
train_out_folder = './Output/emidec_dataset/train/'
test_out_folder = './Output/emidec_dataset/test/'
#---------------------------------------------------------------------------------------------------------------------------------------

#Getting all images from folders
def get_all_images(folder, ext):
    all_files = []
    for file in os.listdir(folder):
        _,  file_ext = os.path.splitext(file)
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)
    return all_files

#-----------------------------------------------------------------------------------

def command_iteration(filter):
    global metric_values
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    metric_values.append(filter.GetMetric())

#-----------------------------------------------------------------------------------

metric_values = []
def non_rigid_registraion(fixed_image, moving_image):
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(5000)
    demons.SetStandardDeviations(1.2)
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

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
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving_image)

    return out

#-----------------------------------------------------------------------------------


metric_values = []
def non_rigid_registraion_paired(fixed_image, moving_mask_file, moving_image_file):
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(5000)
    demons.SetStandardDeviations(1.2)
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                     (sitk.Transform(3, sitk.sitkIdentity))
                                                     )
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)

    displacementField = transform_to_displacment_field_filter.Execute(displacementTx)
    displacementField = demons.Execute(fixed_image, moving_mask_file, displacementField)
    outTx = sitk.DisplacementFieldTransform(displacementField)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    resampler1 = sitk.ResampleImageFilter()
    resampler1.SetReferenceImage(fixed_image)
    resampler1.SetInterpolator(sitk.sitkBSpline)
    resampler1.SetDefaultPixelValue(0)
    resampler1.SetTransform(outTx)

    out_mask = resampler.Execute(moving_mask_file)
    out_image = resampler.Execute(moving_image_file)

    return out_mask, out_image


#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
train_images = []
train_masks = []
for folder_path in os.listdir(train_folder):
    image_file_path = os.path.join(train_folder, folder_path + '/Images')
    mask_file_path = os.path.join(train_folder, folder_path + '/Contours')
    train_images.append(get_all_images(image_file_path, 'gz')[0])
    train_masks.append(get_all_images(mask_file_path, 'gz')[0])
#---------------------------------------------------------------------------
test_images = []
for folder_path in os.listdir(test_folder):
    image_file_path = os.path.join(test_folder, folder_path + '/Images')
    test_images.append(get_all_images(image_file_path, 'gz')[0])
#---------------------------------------------------------------------------
train_images.sort()
train_masks.sort()
test_images.sort()
print(len(train_images))
print(len(train_masks))
print(len(test_images))
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------


for i in range(len(train_masks)):
    moving_mask_file = train_masks[i]
    moving_image_file = train_images[i]
    fixed = sitk.ReadImage("./Template/myo_ED_AHA17.nii.gz", sitk.sitkFloat32)
    moving_mask = sitk.ReadImage(moving_mask_file, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_file, sitk.sitkFloat32)

    out_mask, out_image = non_rigid_registraion_paired(fixed, moving_mask, moving_image)

    new_name = moving_mask_file.split('/')[-1].split('.')[0]
    new_name = new_name.replace('_reg1', '')
    folder_path1 = train_out_folder + new_name + "/Contours/"
    folder_path2 = train_out_folder + new_name + "/Images/"
    if not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    filename1 = folder_path1 + new_name + "_reg2.nii.gz"
    filename2 = folder_path2 + new_name + "_reg2.nii.gz"

    sitk.WriteImage(out_mask, filename1)
    sitk.WriteImage(out_image, filename2)

    print('--------------Successful--------------------')
    print(i, filename1, filename2)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')


#-------------------------------------------------------------------------------------


for i in range(len(test_images)):
    moving_image_file = test_images[i]
    fixed = sitk.ReadImage("./Template/myo_ED_AHA17.nii.gz", sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_file, sitk.sitkFloat32)

    out_image = non_rigid_registraion(fixed, moving_image)

    new_name = moving_image_file.split('/')[-1].split('.')[0]
    new_name = new_name.replace('_reg1', '')
    folder_path2 = test_out_folder + new_name + "/Images/"
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    filename2 = folder_path2 + new_name + "_reg2.nii.gz"

    sitk.WriteImage(out_image, filename2)

    print('--------------Successful--------------------')
    print(i, filename2)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')


#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
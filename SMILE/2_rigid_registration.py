import SimpleITK as sitk
import numpy as np
import nibabel as nib
import os

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
train_folder = './Dataset/emidec_dataset/train'
test_folder = './Dataset/emidec_dataset/test'
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

#-----------------------------------------------------------------------------------------

def command_iteration(method):
    """ Callback invoked when the optimization has an iteration """
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
    )


#-----------------------------------------------------------------------------------------

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
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(compositeTx)
    out = resampler.Execute(moving)

    return out


#-----------------------------------------------------------------------------------------

def rigid_registration_paired(fixed, moving_mask, moving_image):
    initialTx = sitk.CenteredTransformInitializer(fixed, moving_mask, sitk.AffineTransform(fixed.GetDimension()))

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
    outTx1 = R.Execute(fixed, moving_mask)
    displacementField = sitk.TransformToDisplacementFieldFilter()
    displacementField.SetReferenceImage(fixed)
    displacementTx = sitk.DisplacementFieldTransform(displacementField.Execute(sitk.Transform(3, sitk.sitkIdentity)))
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
    R.SetMovingInitialTransform(outTx1)
    R.SetInitialTransform(displacementTx, inPlace=True)

    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.MetricUseFixedImageGradientFilterOff()
    R.Execute(fixed, moving_mask)
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

    resampler1 = sitk.ResampleImageFilter()
    resampler1.SetReferenceImage(fixed)
    resampler1.SetInterpolator(sitk.sitkBSpline)
    resampler1.SetDefaultPixelValue(0)
    resampler1.SetTransform(compositeTx)

    out_mask = resampler.Execute(moving_mask)
    out_image = resampler1.Execute(moving_image)

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

    out_mask, out_image = rigid_registration_paired(fixed, moving_mask, moving_image)

    new_name = moving_mask_file.split('/')[-1].split('.')[0]
    folder_path1 = train_out_folder + new_name + "/Contours/"
    folder_path2 = train_out_folder + new_name + "/Images/"
    if not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    filename1 = folder_path1 + new_name + "_reg1.nii.gz"
    filename2 = folder_path2 + new_name + "_reg1.nii.gz"

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

    out_image = rigid_registration(fixed, moving_image)

    new_name = moving_image_file.split('/')[-1].split('.')[0]
    folder_path2 = test_out_folder + new_name + "/Images/"
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    filename2 = folder_path2 + new_name + "_reg1.nii.gz"

    sitk.WriteImage(out_image, filename2)

    print('--------------Successful--------------------')
    print(i, filename2)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')


#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
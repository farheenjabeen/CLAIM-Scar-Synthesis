import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import nibabel as nib
import os
import numpy as np
import numpy as np
import scipy as sp
import SimpleITK as sitk
import random
import cv2
import shutil
import torch
import torchio as tio


#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def main():

    m = 0
    model_name = ['Output_Lefusion_J_ScarSynth', 'Output_Lefusion_ScarSynth_LeSegLoss', 'Output_Lefusion_ScarSynth']

    in_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/' + model_name[m] + '/generate_slices_from_Normal_51/'
    out_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/' + model_name[m] + '/generate_slices_from_Normal_5/'
    #get_correction(in_folder, out_folder)


    temp_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/train_emidec_dataset/Normal/'
    in_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/'+model_name[m]+'/generate_slices_from_Normal_1/'
    out_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/'+model_name[m]+'/generate_from_Normal_1/'
    #get_post_processed_data_N1(in_folder, temp_folder, out_folder)

    temp_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/train_emidec_dataset/Normal/'
    in_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/'+model_name[m]+'/generate_slices_from_Normal_11/'
    out_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/'+model_name[m]+'/generate_from_Normal_11/'
    #get_post_processed_data_N11(in_folder, temp_folder, out_folder)

    temp_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/train_emidec_dataset/Pathological_1/'
    in_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/'+model_name[m]+'/generate_slices_from_Pathological_1/'
    out_folder = '/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/data/EMIDEC/Outputs/'+model_name[m]+'/generate_from_Pathological_1/'
    #get_post_processed_data_P1(in_folder, temp_folder, out_folder)

#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

#Getting all images from folders
def get_all_images(folder, ext):
    all_files = []
    for file in os.listdir(folder):
        _,  file_ext = os.path.splitext(file)
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)
    return all_files


def normalize_image(image):
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    return image

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def get_correction(in_folder, out_folder):
    full_path1 = in_folder + '/images/'
    full_path2 = in_folder + '/labels/'
    input_images = get_all_images(full_path1, 'gz')
    input_masks = get_all_images(full_path2, 'gz')
    for i in range(len(input_images)):
        name1 = input_images[i].replace('_1_', '_5_')
        name2 = input_masks[i].replace('_1_', '_5_')
        name1 = name1.replace('_51', '_5')
        name2 = name2.replace('_51', '_5')
        shutil.copy(input_images[i], name1)
        shutil.copy(input_masks[i], name2)
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def get_post_processed_data_N1(in_folder, temp_folder, out_folder):
    full_path1 = temp_folder + '/images/'
    full_path2 = temp_folder + '/labels/'
    path_data_images = get_all_images(full_path1, 'gz')
    path_data_masks = get_all_images(full_path2, 'gz')
    path_data_masks.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    path_data_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    full_path1 = in_folder + '/images/'
    full_path2 = in_folder + '/labels/'
    input_images = get_all_images(full_path1, 'gz')
    input_masks = get_all_images(full_path2, 'gz')
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    out_folder_images = out_folder + '/images/'
    out_folder_masks = out_folder + '/labels/'
    if not os.path.exists(out_folder_images):
        os.makedirs(out_folder_images)
    if not os.path.exists(out_folder_masks):
        os.makedirs(out_folder_masks)
    unique_names = []
    for i in range(len(input_images)):
        unique_names.append(input_images[i].split('/')[-1].split('_')[-3])
    unique_names = list(set(unique_names))
    unique_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i in range(len(unique_names)):
        orig_image = nib.load(path_data_images[i]).get_fdata()
        orig_mask = nib.load(path_data_masks[i]).get_fdata()
        # -------------------------------
        slices1 = [x for x in input_images if 'Case_'+unique_names[i]+'_1_' in x]
        mslices1 = [x for x in input_masks if 'Case_'+unique_names[i]+'_1_' in x]
        slices1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        mslices1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image1 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
        mask1 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2])).astype('uint16')
        #-------------------------------
        slices2 = [x for x in input_images if 'Case_'+unique_names[i]+'_2_' in x]
        mslices2 = [x for x in input_masks if 'Case_'+unique_names[i]+'_2_' in x]
        slices2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        mslices2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image2 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
        mask2 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2])).astype('uint16')
        # -------------------------------
        for j in range(orig_image.shape[2]):
            x1 = int(np.ceil((144 - orig_image.shape[0]) / 2))
            x2 = x1 + orig_image.shape[0]
            y1 = int(np.ceil((144 - orig_image.shape[1]) / 2))
            y2 = y1 + orig_image.shape[1]
            image1[:,:,j] = np.reshape(nib.load(slices1[j]).get_fdata(), (144,144))[x1:x2, y1:y2]
            mask1[:,:,j] = np.reshape(nib.load(mslices1[j]).get_fdata(),(144,144))[x1:x2, y1:y2]
            image2[:,:,j]  = np.reshape(nib.load(slices2[j]).get_fdata(), (144,144))[x1:x2, y1:y2]
            mask2[:,:,j] = np.reshape(nib.load(mslices2[j]).get_fdata(), (144,144))[x1:x2, y1:y2]
            '''
            # Visualize results
            corr_img = (normalize_image(orig_image[:,:,j]) - image1[:, :, j])
            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
            ax1, ax2, ax3, ax4 = axes.flatten()
            s11 = ax1.imshow(normalize_image(orig_image[:,:,j]), cmap="gray")
            ax1.set_title('Original Image (Normal)')
            fig.colorbar(s11, ax=ax1, fraction=0.046, pad=0.04)
            s22 = ax2.imshow(image1[:, :, j], cmap="gray")
            ax2.set_title('Generated Image (Pathological)')
            fig.colorbar(s22, ax=ax2, fraction=0.046, pad=0.04)
            s3 = ax3.imshow(corr_img, cmap='RdBu_r', vmax=-np.min(corr_img), interpolation='none')
            ax3.set_title('Comparison')
            fig.colorbar(s3, ax=ax3, fraction=0.046, pad=0.04)
            s4 = ax4.imshow(mask1[:, :, j])
            ax4.set_title('Input Mask')
            fig.colorbar(s4, ax=ax4, fraction=0.046, pad=0.04)
            plt.show()
            '''
        img1 = nib.Nifti1Image(image1, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        msk1 = nib.Nifti1Image(mask1, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        img2 = nib.Nifti1Image(image2, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        msk2 = nib.Nifti1Image(mask2, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        nib.save(img1, out_folder_images + 'Case_'+unique_names[i]+'_1.nii.gz')
        nib.save(msk1, out_folder_masks + 'Case_'+unique_names[i]+'_1.nii.gz')
        nib.save(img2, out_folder_images + 'Case_'+unique_names[i]+'_2.nii.gz')
        nib.save(msk2, out_folder_masks + 'Case_'+unique_names[i]+'_2.nii.gz')
        print(str(i + 1) + ' Processing:', unique_names[i])
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def get_post_processed_data_N11(in_folder, temp_folder, out_folder):
    full_path1 = temp_folder + '/images/'
    full_path2 = temp_folder + '/labels/'
    path_data_images = get_all_images(full_path1, 'gz')
    path_data_masks = get_all_images(full_path2, 'gz')
    path_data_masks.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    path_data_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    full_path1 = in_folder + '/images/'
    full_path2 = in_folder + '/labels/'
    input_images = get_all_images(full_path1, 'gz')
    input_masks = get_all_images(full_path2, 'gz')
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    out_folder_images = out_folder + '/images/'
    out_folder_masks = out_folder + '/labels/'
    if not os.path.exists(out_folder_images):
        os.makedirs(out_folder_images)
    if not os.path.exists(out_folder_masks):
        os.makedirs(out_folder_masks)
    unique_names = []
    for i in range(len(input_images)):
        unique_names.append(input_images[i].split('/')[-1].split('_')[-3])
    unique_names = list(set(unique_names))
    unique_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i in range(len(unique_names)):
        orig_image = nib.load(path_data_images[i]).get_fdata()
        orig_mask = nib.load(path_data_masks[i]).get_fdata()
        # -------------------------------
        slices1 = [x for x in input_images if 'Case_'+unique_names[i]+'_3_' in x]
        mslices1 = [x for x in input_masks if 'Case_'+unique_names[i]+'_3_' in x]
        slices1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        mslices1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image1 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
        mask1 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2])).astype('uint16')
        #-------------------------------
        slices2 = [x for x in input_images if 'Case_'+unique_names[i]+'_4_' in x]
        mslices2 = [x for x in input_masks if 'Case_'+unique_names[i]+'_4_' in x]
        slices2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        mslices2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image2 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
        mask2 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2])).astype('uint16')
        # -------------------------------
        slices3 = [x for x in input_images if 'Case_' + unique_names[i] + '_5_' in x]
        mslices3 = [x for x in input_masks if 'Case_' + unique_names[i] + '_5_' in x]
        slices3.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        mslices3.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image3 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
        mask3 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2])).astype('uint16')
        # -------------------------------
        for j in range(orig_image.shape[2]):
            x1 = int(np.ceil((144 - orig_image.shape[0]) / 2))
            x2 = x1 + orig_image.shape[0]
            y1 = int(np.ceil((144 - orig_image.shape[1]) / 2))
            y2 = y1 + orig_image.shape[1]
            image1[:, :, j] = np.reshape(nib.load(slices1[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            mask1[:, :, j] = np.reshape(nib.load(mslices1[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            image2[:, :, j] = np.reshape(nib.load(slices2[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            mask2[:, :, j] = np.reshape(nib.load(mslices2[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            image3[:, :, j] = np.reshape(nib.load(slices3[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            mask3[:, :, j] = np.reshape(nib.load(mslices3[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            '''
            # Visualize results
            corr_img = (normalize_image(orig_image[:,:,j]) - image1[:, :, j])
            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
            ax1, ax2, ax3, ax4 = axes.flatten()
            s11 = ax1.imshow(normalize_image(orig_image[:,:,j]), cmap="gray")
            ax1.set_title('Original Image (Normal)')
            fig.colorbar(s11, ax=ax1, fraction=0.046, pad=0.04)
            s22 = ax2.imshow(image1[:, :, j], cmap="gray")
            ax2.set_title('Generated Image (Pathological)')
            fig.colorbar(s22, ax=ax2, fraction=0.046, pad=0.04)
            s3 = ax3.imshow(corr_img, cmap='RdBu_r', vmax=-np.min(corr_img), interpolation='none')
            ax3.set_title('Comparison')
            fig.colorbar(s3, ax=ax3, fraction=0.046, pad=0.04)
            s4 = ax4.imshow(mask1[:, :, j])
            ax4.set_title('Input Mask')
            fig.colorbar(s4, ax=ax4, fraction=0.046, pad=0.04)
            plt.show()
            '''
        img1 = nib.Nifti1Image(image1, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        msk1 = nib.Nifti1Image(mask1, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        img2 = nib.Nifti1Image(image2, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        msk2 = nib.Nifti1Image(mask2, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        img3 = nib.Nifti1Image(image3, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        msk3 = nib.Nifti1Image(mask3, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        nib.save(img1, out_folder_images + 'Case_'+unique_names[i]+'_3.nii.gz')
        nib.save(msk1, out_folder_masks + 'Case_'+unique_names[i]+'_3.nii.gz')
        nib.save(img2, out_folder_images + 'Case_'+unique_names[i]+'_4.nii.gz')
        nib.save(msk2, out_folder_masks + 'Case_'+unique_names[i]+'_4.nii.gz')
        nib.save(img3, out_folder_images + 'Case_' + unique_names[i] + '_5.nii.gz')
        nib.save(msk3, out_folder_masks + 'Case_' + unique_names[i] + '_5.nii.gz')
        print(str(i + 1) + ' Processing:', unique_names[i])
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def get_post_processed_data_P1(in_folder, temp_folder, out_folder):
    full_path1 = temp_folder + '/images/'
    full_path2 = temp_folder + '/labels_comb/'
    path_data_images = get_all_images(full_path1, 'gz')
    path_data_masks = get_all_images(full_path2, 'gz')
    path_data_masks.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    path_data_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    full_path1 = in_folder + '/images/'
    full_path2 = in_folder + '/labels/'
    input_images = get_all_images(full_path1, 'gz')
    input_masks = get_all_images(full_path2, 'gz')
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    out_folder_images = out_folder + '/images/'
    out_folder_masks = out_folder + '/labels/'
    if not os.path.exists(out_folder_images):
        os.makedirs(out_folder_images)
    if not os.path.exists(out_folder_masks):
        os.makedirs(out_folder_masks)
    unique_names = []
    for i in range(len(input_images)):
        unique_names.append(input_images[i].split('/')[-1].split('_')[-3])
    unique_names = list(set(unique_names))
    unique_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i in range(len(unique_names)):
        orig_image = nib.load(path_data_images[i]).get_fdata()
        orig_mask = nib.load(path_data_masks[i]).get_fdata()
        # -------------------------------
        slices1 = [x for x in input_images if 'Case_'+unique_names[i]+'_1_' in x]
        mslices1 = [x for x in input_masks if 'Case_'+unique_names[i]+'_1_' in x]
        slices1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        mslices1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image1 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
        mask1 = np.zeros((orig_image.shape[0],orig_image.shape[1],orig_image.shape[2])).astype('uint16')
        # -------------------------------
        for j in range(orig_image.shape[2]):
            x1 = int(np.ceil((144 - orig_image.shape[0]) / 2))
            x2 = x1 + orig_image.shape[0]
            y1 = int(np.ceil((144 - orig_image.shape[1]) / 2))
            y2 = y1 + orig_image.shape[1]
            image1[:, :, j] = np.reshape(nib.load(slices1[j]).get_fdata(), (144, 144))[x1:x2, y1:y2]
            mask1[:, :, j] = orig_mask[:, :, j]
            '''
            # Visualize results
            corr_img = (normalize_image(orig_image[:,:,j]) - image1[:, :, j])
            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
            ax1, ax2, ax3, ax4 = axes.flatten()
            s11 = ax1.imshow(normalize_image(orig_image[:,:,j]), cmap="gray")
            ax1.set_title('Original Image (Normal)')
            fig.colorbar(s11, ax=ax1, fraction=0.046, pad=0.04)
            s22 = ax2.imshow(image1[:, :, j], cmap="gray")
            ax2.set_title('Generated Image (Pathological)')
            fig.colorbar(s22, ax=ax2, fraction=0.046, pad=0.04)
            s3 = ax3.imshow(corr_img, cmap='RdBu_r', vmax=-np.min(corr_img), interpolation='none')
            ax3.set_title('Comparison')
            fig.colorbar(s3, ax=ax3, fraction=0.046, pad=0.04)
            s4 = ax4.imshow(mask1[:, :, j])
            ax4.set_title('Input Mask')
            fig.colorbar(s4, ax=ax4, fraction=0.046, pad=0.04)
            plt.show()
            '''
        img1 = nib.Nifti1Image(image1, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        msk1 = nib.Nifti1Image(mask1, affine=nib.load(path_data_images[i]).affine, header=nib.load(path_data_images[i]).header)
        nib.save(img1, out_folder_images + 'Case_'+unique_names[i]+'_1.nii.gz')
        nib.save(msk1, out_folder_masks + 'Case_'+unique_names[i]+'_1.nii.gz')
        print(str(i + 1) + ' Processing:', unique_names[i])
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
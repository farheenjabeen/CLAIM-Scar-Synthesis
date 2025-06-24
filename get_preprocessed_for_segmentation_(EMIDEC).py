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
from pymic.util.image_process import get_ND_bounding_box
from scipy import ndimage
from skimage.transform import resize
from skimage import exposure


#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def main():
    in_folder = '/home/farheen/E/Scar_Syhthesis/AHA_Analysis/Dataset/emidec_dataset/valid/'
    out_folder = './data/EMIDEC/valid_preprocessed_emidec_dataset/Pathological/'
    get_preprocessed_data_for_segmentation_P(in_folder, out_folder)
    
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

w, h = 160, 160

#Getting all images from folders
def get_all_images(folder, ext):
    all_files = []
    for file in os.listdir(folder):
        _,  file_ext = os.path.splitext(file)
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)
    return all_files

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def get_preprocessed_data_for_segmentation_P(data_folder, out_folder):
    image_files = []
    mask_files = []
    for folder_path in os.listdir(data_folder):
        if ('Case_P' in folder_path):
            mask_file_path = os.path.join(data_folder, folder_path + '/Contours')
            image_file_path = os.path.join(data_folder, folder_path + '/Images')
            mask_files.append(get_all_images(mask_file_path, 'gz'))
            image_files.append(get_all_images(image_file_path, 'gz'))
        else:
            continue
    path_data_masks = []
    path_data_images = []
    for i in range(len(mask_files)):
        gd = mask_files[i]
        image = image_files[i]
        path_data_masks.append(gd[0])
        path_data_images.append(image[0])
    path_data_masks.sort()
    path_data_images.sort()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    full_path1 = out_folder + '/images/'
    full_path2 = out_folder + '/labels/'
    if not os.path.exists(full_path1):
        os.makedirs(full_path1)
    if not os.path.exists(full_path2):
        os.makedirs(full_path2)
    for f in range(len(path_data_masks)):
        image = nib.load(path_data_images[f]).get_fdata()
        mask = nib.load(path_data_masks[f]).get_fdata()
        resize_image = []
        resize_mask  = []
        for j in range(mask.shape[2]):
            img = resize(image[:,:,j], (256, 256))
            msk = cv2.resize(mask[:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)
            resize_image.append(img)
            resize_mask.append(msk)
        image = np.moveaxis(np.array(resize_image), 0, -1)
        mask = np.moveaxis(np.array(resize_mask), 0, -1)
        image_data, mask_data = crop_from_center(image, mask, w, h)
        mask_data[mask_data == 4] =  2
        mask = nib.Nifti1Image(mask_data.astype('int64'), affine=nib.load(path_data_masks[f]).affine, header=nib.load(path_data_masks[f]).header)
        img = nib.Nifti1Image(image_data.astype('int64'), affine=nib.load(path_data_masks[f]).affine, header=nib.load(path_data_masks[f]).header)
        name = path_data_images[f].split('/')[-1].split('.')[-3]
        nib.save(mask, full_path2 + name + '.nii.gz')
        nib.save(img, full_path1 + name + '.nii.gz')
        print(str(f + 1) + ' Processing:', name, np.shape(image_data), np.shape(mask_data))
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def crop_from_center(input_image, input_mask, w, h):
        width, height = w, h
        margin = [20, 20, 20]
        out_image = []
        out_mask = []
        # Crop from center
        crop_bbox_min, crop_bbox_max = get_ND_bounding_box(input_mask)
        y_offset = ((crop_bbox_max[1] - crop_bbox_min[1]) - height)
        x_offset = ((crop_bbox_max[0] - crop_bbox_min[0]) - width)
        if((x_offset% 2) == 0):
            x1_offset = int(x_offset/2)
            x2_offset = int(x_offset/2)
        else:
            x1_offset = int(x_offset/2)+1
            x2_offset = int(x_offset/2)
        if((y_offset % 2) == 0):
            y1_offset = int(y_offset/2)
            y2_offset = int(y_offset/2)
        else:
            y1_offset = int(y_offset/2)+1
            y2_offset = int(y_offset/2)
        for j in range(input_mask.shape[2]):
            out_image.append(input_image[crop_bbox_min[0]+x1_offset:crop_bbox_max[0]-x2_offset,crop_bbox_min[1]+y1_offset:crop_bbox_max[1]-y2_offset,j])
            out_mask.append(input_mask[crop_bbox_min[0]+x1_offset:crop_bbox_max[0]-x2_offset,crop_bbox_min[1]+y1_offset:crop_bbox_max[1]-y2_offset,j])
        output_image = np.moveaxis(np.array(out_image), 0, -1)
        output_mask = np.moveaxis(np.array(out_mask), 0, -1)
        return output_image, output_mask

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
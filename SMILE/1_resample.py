import nibabel as nib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2

#---------------------------------------------------------------------------------------------------------------------------------------
train_folder = './Dataset/emidec_dataset/train'
test_folder = './Dataset/emidec_dataset/test'
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

for i in range(len(train_images)):
    file_image = train_images[i]
    from_image =  nib.load(file_image)
    from_temp =  nib.load("./Template/myo_ED_AHA17.nii.gz") #AHA17_model_ED.nii.gz
    image = from_image.get_fdata()
    temp = from_temp.get_fdata()

    x1, y1, z1 = nib.aff2axcodes(from_image.affine)
    #print('Subject:', x1,y1,z1)
    x2, y2, z2 = nib.aff2axcodes(from_temp.affine)
    #print('Template', x2,y2,z2)

    #print(from_temp.affine)
    #print('---------------------------')
    #print(from_image.affine)
    #print('---------------------------')

    img_affine = from_image.affine
    if(x1=="P" and y1=="I" and z1=="L"):
        img_affine[0,0] = img_affine[1,0]
        img_affine[1,0] = 0.0
        img_affine[1,1] = -img_affine[2,1]
        img_affine[2,1] = 0.0
        img_affine[2,2] = img_affine[0,2]
        img_affine[0,2] = 0.0
        print('processing.....')
    elif(x1 == "L" and y1 == "I" and z1 == "P"):
        img_affine[2, 2] = img_affine[1, 2]
        img_affine[1, 2] = 0.0
        img_affine[1, 1] = img_affine[2, 1]
        img_affine[2, 1] = 0.
        print('processing.....')
    elif(x1 == "L" and y1 == "P" and z1 == "S"):
        continue


#----------------------------------------------------------------------------

for i in range(len(test_images)):
    file_image = test_images[i]
    from_image =  nib.load(file_image)
    from_temp =  nib.load("./Template/myo_ED_AHA17.nii.gz") #AHA17_model_ED.nii.gz
    image = from_image.get_fdata()
    temp = from_temp.get_fdata()

    x1, y1, z1 = nib.aff2axcodes(from_image.affine)
    #print('Subject:', x1,y1,z1)
    x2, y2, z2 = nib.aff2axcodes(from_temp.affine)
    #print('Template', x2,y2,z2)

    #print(from_temp.affine)
    #print('---------------------------')
    #print(from_image.affine)
    #print('---------------------------')

    img_affine = from_image.affine
    if(x1=="P" and y1=="I" and z1=="L"):
        img_affine[0,0] = img_affine[1,0]
        img_affine[1,0] = 0.0
        img_affine[1,1] = -img_affine[2,1]
        img_affine[2,1] = 0.0
        img_affine[2,2] = img_affine[0,2]
        img_affine[0,2] = 0.0
        print('processing.....')
    elif(x1 == "L" and y1 == "I" and z1 == "P"):
        img_affine[2, 2] = img_affine[1, 2]
        img_affine[1, 2] = 0.0
        img_affine[1, 1] = img_affine[2, 1]
        img_affine[2, 1] = 0.
        print('processing.....')
    elif(x1 == "L" and y1 == "P" and z1 == "S"):
        continue

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------


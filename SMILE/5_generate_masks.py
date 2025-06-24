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
data_id = 'Nv3'
data_folder = './Output/emidec_dataset/valid/'
template_img_path = "./Template/myo_ED_AHA17.nii.gz"
out_folder = './GenMasks1/'+data_id+'/emidec_dataset/train/'
Bulls_eye_plots_path_Vol = './GenMasks1/'+data_id+'/emidec_dataset/train_vol_plots'
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def main():
    generate_from_normal(data_folder, out_folder, template_img_path, max_scar_vol = 12.0) # 6.0 (old value)
    #generate_from_pathology(data_folder, out_folder, template_img_path, max_scar_vol = 9.0) # 3.0 (old value)

    train_files = []
    for folder_path in os.listdir(out_folder):
        if('Case' in folder_path):
            mask_file_path = os.path.join(out_folder, folder_path + '/Contours')
            train_files.append(get_all_images(mask_file_path, 'gz'))
        else:
            continue
    train_masks = []
    for i in range(len(train_files)):
        if ('Case_N' in train_files[i][0]):
            gd = [x for x in train_files[i] if ('_scar' not in x)]
        elif('Case_P' in train_files[i][0]):
            gd = [x for x in train_files[i] if ('_scar' not in x and '_comb' not in x  and '_orig' not in x)]
        train_masks.append(gd[0])
    train_masks.sort()
    #--------------------------------------------------------------------------------
    #fit_average_bulleye_plot1(train_masks, template_img_path)
    for i in range(len(train_masks)):
        print(train_masks[i])
        fit_average_bulleye_plot([train_masks[i]], template_img_path, Bulls_eye_plots_path_Vol)

#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------


def generate_from_normal(data_folder, out_folder, template_img_path, max_scar_vol):
    image_files = []
    mask_files = []
    for folder_path in os.listdir(data_folder):
        if('Case_N' in folder_path):
            mask_file_path = os.path.join(data_folder, folder_path + '/Contours')
            image_file_path = os.path.join(data_folder, folder_path + '/Images')
            mask_files.append(get_all_images(mask_file_path, 'gz'))
            image_files.append(get_all_images(image_file_path, 'gz'))
        else:
            continue
    data_masks = []
    data_images = []
    for i in range(len(mask_files)):
        gd = [x for x in mask_files[i] if 'reg2' in x]
        image = [x for x in image_files[i] if 'reg2' in x]
        data_masks.append(gd[0])
        data_images.append(image[0])
    data_masks.sort()
    data_images.sort()
    #--------------------------------------------------------------------------------
    sigma = 1.0
    threshold = 0.3
    for f in range(len(data_masks)):
        scar_volumes = sample_floats(3.0, max_scar_vol, k=17)
        regions = select_aha_regions()
        out_blobs = generate_mask(data_masks[f], template_img_path, scar_Volumes=scar_volumes, regions=regions)
        filled_mask = binary_fill_holes(np.array(out_blobs, dtype=int))
        smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)
        smoothed_mask_binary = smoothed_mask > threshold
        labeled_mask, num_features = label(smoothed_mask_binary)
        region_sizes = np.bincount(labeled_mask.ravel())
        region_sizes[0] = 0
        largest_region_label = region_sizes.argmax()
        largest_region_mask = (labeled_mask == largest_region_label)
        #mask1 = nib.Nifti1Image(out_blobs.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        mask1 = nib.Nifti1Image(largest_region_mask.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        name = data_masks[f].split('/')[-1].split('.')[-3]
        name = name.replace('_reg2', '')
        full_path1 = out_folder + name + '/Contours/'
        if not os.path.exists(full_path1):
            os.makedirs(full_path1)
        nib.save(mask1, full_path1 + name + '_scar.nii.gz')

        mask_data = nib.load(data_masks[f]).get_fdata()
        mask_data[mask_data == 1] = 1
        mask_data[mask_data == 2] = 2
        mask_data[mask_data == 3] = 2
        mask_data[mask_data == 4] = 2
        mask_data[largest_region_mask == 1] = 3
        mask2 = nib.Nifti1Image(mask_data.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        nib.save(mask2, full_path1 + name + '.nii.gz')

        full_path2 = out_folder + name + '/Images/'
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        source = data_images[f]
        dest = full_path2 + name + '.nii.gz'
        shutil.copy(source, dest)
        print(str(f+1) + ' Processing:', name)
    return



def generate_from_normal1(data_folder, out_folder, template_img_path, max_scar_vol):
    image_files = []
    mask_files = []
    for folder_path in os.listdir(data_folder):
        if('Case_N' in folder_path):
            mask_file_path = os.path.join(data_folder, folder_path + '/Contours')
            image_file_path = os.path.join(data_folder, folder_path + '/Images')
            mask_files.append(get_all_images(mask_file_path, 'gz'))
            image_files.append(get_all_images(image_file_path, 'gz'))
        else:
            continue
    data_masks = []
    data_images = []
    for i in range(len(mask_files)):
        gd = [x for x in mask_files[i] if 'reg2' in x]
        image = [x for x in image_files[i] if 'reg2' in x]
        data_masks.append(gd[0])
        data_images.append(image[0])
    data_masks.sort()
    data_images.sort()
    #--------------------------------------------------------------------------------
    sigma = 1.0
    threshold = 0.3
    for f in range(len(data_masks)):
        scar_volumes = sample_floats(3.0, max_scar_vol, k=17)
        regions = select_aha_regions()
        out_blobs = generate_mask(data_masks[f], template_img_path, scar_Volumes=scar_volumes, regions=regions)

        filled_mask = binary_fill_holes(np.array(out_blobs, dtype=int))
        smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)
        smoothed_mask_binary = smoothed_mask > threshold
        labeled_mask, num_features = label(smoothed_mask_binary)
        region_sizes = np.bincount(labeled_mask.ravel())
        region_sizes[0] = 0
        largest_region_label = region_sizes.argmax()
        largest_region_mask = (labeled_mask == largest_region_label)
        #mask1 = nib.Nifti1Image(out_blobs.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        mask1 = nib.Nifti1Image(largest_region_mask.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        name = data_masks[f].split('/')[-1].split('.')[-3]
        name = name.replace('_reg2', '')
        full_path1 = out_folder + '/labels/'
        if not os.path.exists(full_path1):
            os.makedirs(full_path1)

        mask_data = nib.load(data_masks[f]).get_fdata()
        mask_data[mask_data == 1] = 1
        mask_data[mask_data == 2] = 2
        mask_data[mask_data == 3] = 2
        mask_data[mask_data == 4] = 2
        mask_data[largest_region_mask == 1] = 3
        mask2 = nib.Nifti1Image(mask_data.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        nib.save(mask2, full_path1 + name + '.nii.gz')

        full_path2 = out_folder + '/images/'
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        source = data_images[f]
        dest = full_path2 + name + '.nii.gz'
        shutil.copy(source, dest)
        print(str(f+1) + ' Processing:', name)
    return



#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def generate_from_pathology(data_folder, out_folder, template_img_path, max_scar_vol):
    image_files = []
    mask_files = []
    for folder_path in os.listdir(data_folder):
        if('Case_P' in folder_path):
            mask_file_path = os.path.join(data_folder, folder_path + '/Contours')
            image_file_path = os.path.join(data_folder, folder_path + '/Images')
            mask_files.append(get_all_images(mask_file_path, 'gz'))
            image_files.append(get_all_images(image_file_path, 'gz'))
        else:
            continue
    data_masks = []
    data_images = []
    for i in range(len(mask_files)):
        gd = [x for x in mask_files[i] if 'reg2' in x]
        image = [x for x in image_files[i] if 'reg2' in x]
        data_masks.append(gd[0])
        data_images.append(image[0])
    data_masks.sort()
    data_images.sort()
    #--------------------------------------------------------------------------------
    sigma = 1.0
    threshold = 0.3
    for f in range(len(data_masks)):
        scar_volumes = sample_floats(3.0, max_scar_vol, k=17)
        regions = select_aha_regions()
        out_blobs = generate_mask(data_masks[f], template_img_path, scar_Volumes=scar_volumes, regions=regions)

        filled_mask = binary_fill_holes(np.array(out_blobs, dtype=int))
        smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)
        smoothed_mask_binary = smoothed_mask > threshold
        labeled_mask, num_features = label(smoothed_mask_binary)
        region_sizes = np.bincount(labeled_mask.ravel())
        region_sizes[0] = 0
        largest_region_label = region_sizes.argmax()
        largest_region_mask = (labeled_mask == largest_region_label)
        #mask1 = nib.Nifti1Image(out_blobs.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        mask1 = nib.Nifti1Image(largest_region_mask.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        name = data_masks[f].split('/')[-1].split('.')[-3]
        name = name.replace('_reg2', '')
        full_path1 = out_folder + name + '/Contours/'
        if not os.path.exists(full_path1):
            os.makedirs(full_path1)
        nib.save(mask1, full_path1 + name + '_scar.nii.gz')

        mask_data = nib.load(data_masks[f]).get_fdata()
        mask_data[mask_data == 1] = 1
        mask_data[mask_data == 2] = 2
        mask_data[mask_data == 3] = 3
        mask_data[mask_data == 4] = 2
        mask_data[largest_region_mask == 1] = 3
        mask2 = nib.Nifti1Image(mask_data.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        nib.save(mask2, full_path1 + name + '_comb.nii.gz')
        
        mask_data = nib.load(data_masks[f]).get_fdata()
        mask_data[mask_data == 1] = 1
        mask_data[mask_data == 2] = 2
        mask_data[mask_data == 3] = 2
        mask_data[mask_data == 4] = 2
        mask_data[largest_region_mask == 1] = 3
        mask2 = nib.Nifti1Image(mask_data.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        nib.save(mask2, full_path1 + name + '.nii.gz')

        source = data_masks[f]
        dest = full_path1 + name + '_orig.nii.gz'
        shutil.copy(source, dest)

        full_path2 = out_folder + name + '/Images/'
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        source = data_images[f]
        dest = full_path2 + name + '.nii.gz'
        shutil.copy(source, dest)
        print(str(f+1) + ' Processing:', name)
    return



def generate_from_pathology1(data_folder, out_folder, template_img_path, max_scar_vol):
    image_files = []
    mask_files = []
    for folder_path in os.listdir(data_folder):
        if('Case_P' in folder_path):
            mask_file_path = os.path.join(data_folder, folder_path + '/Contours')
            image_file_path = os.path.join(data_folder, folder_path + '/Images')
            mask_files.append(get_all_images(mask_file_path, 'gz'))
            image_files.append(get_all_images(image_file_path, 'gz'))
        else:
            continue
    data_masks = []
    data_images = []
    for i in range(len(mask_files)):
        gd = [x for x in mask_files[i] if 'reg2' in x]
        image = [x for x in image_files[i] if 'reg2' in x]
        data_masks.append(gd[0])
        data_images.append(image[0])
    data_masks.sort()
    data_images.sort()
    #--------------------------------------------------------------------------------
    sigma = 1.0
    threshold = 0.3
    for f in range(len(data_masks)):
        scar_volumes = sample_floats(3.0, max_scar_vol, k=17)
        regions = select_aha_regions()
        out_blobs = generate_mask(data_masks[f], template_img_path, scar_Volumes=scar_volumes, regions=regions)

        filled_mask = binary_fill_holes(np.array(out_blobs, dtype=int))
        smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)
        smoothed_mask_binary = smoothed_mask > threshold
        labeled_mask, num_features = label(smoothed_mask_binary)
        region_sizes = np.bincount(labeled_mask.ravel())
        region_sizes[0] = 0
        largest_region_label = region_sizes.argmax()
        largest_region_mask = (labeled_mask == largest_region_label)
        #mask1 = nib.Nifti1Image(out_blobs.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        mask1 = nib.Nifti1Image(largest_region_mask.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        name = data_masks[f].split('/')[-1].split('.')[-3]
        name = name.replace('_reg2', '')
        full_path1 = out_folder + '/labels/'
        if not os.path.exists(full_path1):
            os.makedirs(full_path1)

        mask_data = nib.load(data_masks[f]).get_fdata()
        mask_data[mask_data == 1] = 1
        mask_data[mask_data == 2] = 2
        mask_data[mask_data == 3] = 3
        mask_data[mask_data == 4] = 2
        mask_data[largest_region_mask == 1] = 3
        mask2 = nib.Nifti1Image(mask_data.astype('int64'), affine=nib.load(template_img_path).affine, header=nib.load(template_img_path).header)
        nib.save(mask2, full_path1 + name + '.nii.gz')

        full_path2 = out_folder +  '/images/'
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        source = data_images[f]
        dest = full_path2 + name + '.nii.gz'
        shutil.copy(source, dest)
        print(str(f+1) + ' Processing:', name)
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


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

def sample_floats(low, high, k=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def select_aha_regions():
    basal_regions = [1, 2, 3, 4, 5, 6]
    middle_regions = [7, 8, 9, 10, 11, 12]
    apical_regions = [[13], [13, 14], [14, 15], [15], [15, 16], [13, 16]]
    basal_adj = [[2, 6, 7], [1, 3, 8], [2, 4, 9], [3, 5, 10], [4, 6, 11], [1, 5, 12]]
    middle_adj = [[1, 8, 12, 13], [2, 7, 9, 13, 14], [3, 8, 10, 14, 15], [4, 9, 11, 15], [5, 10, 12, 15, 16], [6, 7, 11, 13, 16]]
    apical_adj = [[7, 8, 12, 14, 16], [8, 9, 13, 15], [9, 10, 11, 14, 16], [11, 12, 13, 15]]

    my_regions = []
    # -------------------------------------------------------------
    b1 = np.random.choice(basal_regions, np.random.choice(basal_regions, 1), replace=False).tolist()
    for n in range(len(b1)):
        my_regions.append(b1[n])
    for j in range(len(b1)):
        b_size = np.random.choice(np.arange(1,len(basal_adj[b1[j]-1])))
        b2 = np.random.choice(basal_adj[b1[j]-1], b_size, replace=False).tolist()
        for n in range(len(b2)):
            my_regions.append(b2[n])
    my_regions = list(dict.fromkeys(my_regions))
    my_regions.sort()
    # -------------------------------------------------------------
    middle_regions2 = []
    for m in range(len(my_regions)):
        if(my_regions[m]<=6):
            middle_regions2.append(middle_regions[my_regions[m]-1])
    middle_regions2.sort()
    m1 = np.random.choice(middle_regions2)
    if(len(my_regions)!= 0 and [x for x in basal_adj if m1 in x]):
        my_regions.append(m1)
    m_size = np.random.choice(np.arange(1,len(middle_adj[m1-7])))
    m2 = np.random.choice(middle_adj[m1-7], m_size, replace=False).tolist()
    for n in range(len(m2)):
        if (len(my_regions) != 0 and [x for x in basal_adj if m1 in x]):
            my_regions.append(m2[n])
    # -------------------------------------------------------------
    apical_regions2 = []
    for m in range(len(my_regions)):
        if (my_regions[m] <= 12):
            values = apical_regions[my_regions[m] - 7]
            for t in range(len(values)):
                apical_regions2.append(values[t])
    apical_regions2 = list(dict.fromkeys(apical_regions2))
    apical_regions2.sort()
    a1 = np.random.choice(apical_regions2)
    my_regions.append(a1)
    a_size = np.random.choice(np.arange(1,len(apical_adj[a1-13])))
    a2 = np.random.choice(apical_adj[a1-13], a_size, replace=False).tolist()
    for n in range(len(a2)):
        my_regions.append(a2[n])
    #-------------------------------------------------------------
    my_regions = list(dict.fromkeys(my_regions))
    my_regions.sort()
    regions = np.zeros((17))
    for r1 in range(1, len(regions) + 1):
        for r2 in range(len(my_regions)):
            if (r1 == my_regions[r2]):
                regions[r1 - 1] = r1
    return regions

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def generate_blob(subject_img_path, template_img_path, region, scar_Volume, kernel):
    sub_img = nib.load(subject_img_path)
    temp_img = nib.load(template_img_path)
    sub_img_data = sub_img.get_fdata()
    temp_img_data = temp_img.get_fdata()

    sub_img_data[sub_img_data == 1] = 0
    sub_img_data[sub_img_data == 2] = 1
    sub_img_data[sub_img_data == 3] = 1
    sub_img_data[sub_img_data == 4] = 1

    atlas = np.zeros((temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2], 1))
    atlas[temp_img_data == region] = 1
    atlas = np.resize(atlas, (temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2]))

    blob = np.zeros((temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2])).astype('uint16')
    im = ps.generators.blobs(shape=[temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2]], porosity=0.21*scar_Volume, blobiness=1, seed=1001)
    blob[im == True] = 1.0
    kernels = [np.ones((1, 1), np.uint8), np.ones((3, 3), np.uint8),
               np.ones((5, 5), np.uint8), np.ones((7, 7), np.uint8)]
    blob = cv2.erode(blob, kernels[kernel])
    out = sub_img_data * atlas * blob
    return out

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def generate_mask(subject_img_path, template_img_path, scar_Volumes, regions):
    temp_img = nib.load(template_img_path)
    temp_img_data = temp_img.get_fdata()
    #regions = np.unique(temp_img_data)[1::]
    k = np.random.choice(4,17, replace=True)

    out_blobs = np.zeros((temp_img_data.shape[0], temp_img_data.shape[1], temp_img_data.shape[2]))
    for i in range(0, len(regions)):
        if(regions[i] == 0):
            continue
        else:
            blob = generate_blob(subject_img_path, template_img_path, regions[i], scar_Volumes[i], k[i])
            out_blobs = out_blobs + blob
    return out_blobs

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

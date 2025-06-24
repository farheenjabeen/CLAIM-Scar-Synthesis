from pickletools import uint8

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
from PIL import Image
from PIL import ImageChops
from skimage import exposure
from matplotlib.colors import LinearSegmentedColormap
from pymic.util.image_process import get_ND_bounding_box

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def main():
    data_id = 'Normal_1'
    orig_folder = './data/EMIDEC/train_emidec_dataset_slices/'+data_id+'/'
    gen_folder1 = './data/EMIDEC/Outputs/Output_Lefusion_ScarSynth/generate_slices_from_'+data_id+'/'
    gen_folder2 = './data/EMIDEC/Outputs/Output_Lefusion_ScarSynth_LeSegLoss/generate_slices_from_'+data_id+'/'
    gen_folder3 = './data/EMIDEC/Outputs/Output_Lefusion_J_ScarSynth/generate_slices_from_'+data_id+'/'
    out_folder = './data/EMIDEC/Outputs/Plots_Comparison/generate_slices_from_'+data_id+'/'

    evaluate_generated_slices(orig_folder, gen_folder1, gen_folder2, gen_folder3, out_folder)

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

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


def evaluate_generated_slices(orig_folder, gen_folder1, gen_folder2, gen_folder3, out_folder):
    mask_file_path = os.path.join(orig_folder + '/labels')
    orig_image_file_path = os.path.join(orig_folder + '/images')
    gen_image_file_path1 = os.path.join(gen_folder1 + '/images')
    gen_image_file_path2 = os.path.join(gen_folder2 + '/images')
    gen_image_file_path3 = os.path.join(gen_folder3 + '/images')
    mask_files = get_all_images(mask_file_path, 'gz')
    orig_image_files = get_all_images(orig_image_file_path, 'gz')
    gen_image_files1 = get_all_images(gen_image_file_path1, 'gz')
    gen_image_files2 = get_all_images(gen_image_file_path2, 'gz')
    gen_image_files3 = get_all_images(gen_image_file_path3, 'gz')

    if('Normal_1' in orig_folder):
        gen_image_files1 = [x for x in gen_image_files1 if '_1_' in x]
        gen_image_files2 = [x for x in gen_image_files2 if '_1_' in x]
        gen_image_files3 = [x for x in gen_image_files3 if '_1_' in x]
    if ('Normal_2' in orig_folder):
        gen_image_files1 = [x for x in gen_image_files1 if '_2_' in x]
        gen_image_files2 = [x for x in gen_image_files2 if '_2_' in x]
        gen_image_files3 = [x for x in gen_image_files3 if '_2_' in x]
    if ('Normal_3' in orig_folder):
        gen_image_files1 = [x for x in gen_image_files1 if '_3_' in x]
        gen_image_files2 = [x for x in gen_image_files2 if '_3_' in x]
        gen_image_files3 = [x for x in gen_image_files3 if '_3_' in x]
    if ('Normal_4' in orig_folder):
        gen_image_files1 = [x for x in gen_image_files1 if '_4_' in x]
        gen_image_files2 = [x for x in gen_image_files2 if '_4_' in x]
        gen_image_files3 = [x for x in gen_image_files3 if '_4_' in x]
    if ('Normal_5' in orig_folder):
        gen_image_files1 = [x for x in gen_image_files1 if '_5_' in x]
        gen_image_files2 = [x for x in gen_image_files2 if '_5_' in x]
        gen_image_files3 = [x for x in gen_image_files3 if '_5_' in x]
    if ('Pathological_1' in orig_folder):
        gen_image_files1 = [x for x in gen_image_files1 if '_1_' in x]
        gen_image_files2 = [x for x in gen_image_files2 if '_1_' in x]
        gen_image_files3 = [x for x in gen_image_files3 if '_1_' in x]
        
    mask_files.sort()
    orig_image_files.sort()
    gen_image_files1.sort()
    gen_image_files2.sort()
    gen_image_files3.sort()
    print(len(orig_image_files))
    print(len(mask_files))
    print(len(gen_image_files1))
    print(len(gen_image_files2))
    print(len(gen_image_files3))
    # ------------------------------------------------------------------------
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i in range(len(gen_image_files1)):
        file = gen_image_files1[i].split('/')[-1]
        mask_files[i] = orig_folder+'labels/'+file
        orig_image_files[i] = orig_folder+'images/'+file
        gen_image_files1[i] = gen_folder1+'images/'+file
        gen_image_files2[i] = gen_folder2 + 'images/' + file
        gen_image_files3[i] = gen_folder3 + 'images/' + file

        mask_data = nib.load(mask_files[i]).get_fdata()
        orig_img_data = nib.load(orig_image_files[i]).get_fdata()
        gen_img_data1 = nib.load(gen_image_files1[i]).get_fdata()
        gen_img_data2 = nib.load(gen_image_files2[i]).get_fdata()
        gen_img_data3 = nib.load(gen_image_files3[i]).get_fdata()

        x1 = int(np.ceil((144 - mask_data.shape[0]) / 2))
        x2 = x1 + mask_data.shape[0]
        y1 = int(np.ceil((144 - mask_data.shape[1]) / 2))
        y2 = y1 + mask_data.shape[1]
        gen_img_data1 = gen_img_data1[x1:x2, y1:y2]
        gen_img_data2 = gen_img_data2[x1:x2, y1:y2]
        gen_img_data3 = gen_img_data3[x1:x2, y1:y2]

        mask_data = cv2.rotate(cv2.flip(mask_data, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)
        orig_img_data = cv2.rotate(cv2.flip(orig_img_data, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)
        gen_img_data1 = cv2.rotate(cv2.flip(gen_img_data1, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)
        gen_img_data2 = cv2.rotate(cv2.flip(gen_img_data2, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)
        gen_img_data3 = cv2.rotate(cv2.flip(gen_img_data3, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)

        orig_img_data = (orig_img_data - np.min(orig_img_data)) / (np.max(orig_img_data) - np.min(orig_img_data))
        gen_img_data1 = (gen_img_data1 - np.min(gen_img_data1)) / (np.max(gen_img_data1) - np.min(gen_img_data1))
        gen_img_data2 = (gen_img_data2 - np.min(gen_img_data2)) / (np.max(gen_img_data2) - np.min(gen_img_data2))
        gen_img_data3 = (gen_img_data3 - np.min(gen_img_data3)) / (np.max(gen_img_data3) - np.min(gen_img_data3))

        #gen_img_data1 = np.squeeze(gen_img_data1, 2)
        #gen_img_data2 = np.squeeze(gen_img_data2, 2)
        #gen_img_data3 = np.squeeze(gen_img_data3, 2)

        corr_img1 = (orig_img_data - gen_img_data1) * (mask_data == 3)
        corr_img2 = (orig_img_data - gen_img_data2) * (mask_data == 3)
        corr_img3 = (orig_img_data - gen_img_data3) * (mask_data == 3)

        # ----------------------------------------------------------------------------
        # Contrast stretching
        #p1, p2 = np.percentile(orig_img_data, (20,98))
        #orig_img_data = exposure.rescale_intensity(orig_img_data, in_range=(p1, p2))
        #gen_img_data1 = exposure.rescale_intensity(gen_img_data1, in_range=(p1, p2))
        #gen_img_data2 = exposure.rescale_intensity(gen_img_data2, in_range=(p1, p2))
        #gen_img_data3 = exposure.rescale_intensity(gen_img_data3, in_range=(p1, p2))
        #----------------------------------------------------------------------------

        # Extract mask edges
        imgray = mask_data.copy()
        imgray[imgray == 1] = 0
        imgray[imgray == 2] = 0
        imgray[imgray == 3] = 1
        imgray = imgray * 255
        imgray = imgray.astype(np.uint8)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        mask_edges = cv2.Canny(thresh, 0, 255)
        # ----------------------------------------------------------------------------

        fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(10, 20))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

        colors = [(0, 0, 1), (0, 0, 1), (0, 0, 1)]
        my_cmap = LinearSegmentedColormap.from_list('my_list', colors, N=3)

        s1 = ax1.imshow(orig_img_data, cmap="gray")
        ax1.axis('off')
        #ax1.set_title('Original Image (Normal)')
        #fig.colorbar(s1, ax=ax1, fraction=0.046, pad=0.04)

        scar_mask = mask_data.copy()
        scar_mask[scar_mask==1]=0
        scar_mask[scar_mask==2]=0
        scar_mask[scar_mask==3]=1
        s2 = ax2.imshow(scar_mask, cmap='gray')
        #s2 = ax2.imshow(scar_mask, cmap=my_cmap, alpha=1.0*(scar_mask>=1))
        #ax2.set_title('Input Mask')
        #fig.colorbar(s2, ax=ax2, fraction=0.046, pad=0.04)

        s3 = ax3.imshow(gen_img_data1, cmap="gray")
        #s3 = ax3.imshow(mask_edges, cmap=my_cmap, alpha=0.9 * (mask_edges > 1))
        ax3.axis('off')
        #ax3.set_title('Generated Image (DiffScarSynth)')
        #fig.colorbar(s3, ax=ax3, fraction=0.046, pad=0.04)

        s4 = ax4.imshow(corr_img1, cmap='RdBu_r', vmin=np.min(np.array((corr_img1, corr_img2, corr_img3))), vmax=-np.min(np.array((corr_img1, corr_img2, corr_img3))), interpolation='none')
        #ax4.set_title('Comparison (DiffScarSynth)')
        ax4.axis('off')
        fig.colorbar(s4, ax=ax4, fraction=0.046, pad=0.04)

        s5 = ax5.imshow(gen_img_data2, cmap="gray")
        #s5 = ax5.imshow(mask_edges, cmap=my_cmap, alpha=0.9*(mask_edges>1))
        #ax5.set_title('Generated Image (DiffScarSynth+LeSegLoss)')
        ax5.axis('off')
        #fig.colorbar(s5, ax=ax5, fraction=0.046, pad=0.04)

        s6 = ax6.imshow(corr_img2, cmap='RdBu_r', vmin=np.min(np.array((corr_img1, corr_img2, corr_img3))), vmax=-np.min(np.array((corr_img1, corr_img2, corr_img3))), interpolation='none')
        #ax6.set_title('Comparison (DiffScarSynth+LeSegLoss)')
        ax6.axis('off')
        fig.colorbar(s6, ax=ax6, fraction=0.046, pad=0.04)

        s7 = ax7.imshow(gen_img_data3, cmap="gray")
        #s7 = ax7.imshow(mask_edges, cmap=my_cmap, alpha=0.9 * (mask_edges > 1))
        #ax7.set_title('Generated Image (DiffScarSynth+LeSegLoss(J))')
        ax7.axis('off')
        #fig.colorbar(s7, ax=ax7, fraction=0.046, pad=0.04)

        s8 = ax8.imshow(corr_img3, cmap='RdBu_r', vmin=np.min(np.array((corr_img1, corr_img2, corr_img3))), vmax=-np.min(np.array((corr_img1, corr_img2, corr_img3))), interpolation='none')
        #ax8.set_title('Comparison (DiffScarSynth+LeSegLoss(J))')
        ax8.axis('off')
        fig.colorbar(s8, ax=ax8, fraction=0.046, pad=0.04)

        #plt.show()
        fig.savefig(out_folder+file.split('.')[-3]+'.png')
        print(str(i + 1) + ' Processing:', file)
    return

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()



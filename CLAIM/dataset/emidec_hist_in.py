from torch.utils.data import Dataset
import torchio as tio
import os
import torch
import numpy as np

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)), 
    tio.CropOrPad(target_shape=(72, 72, 1))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

class EMIDECInDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.image_paths = self.get_image_files()


    def get_image_files(self):
        nifti_file_names = os.listdir(self.image_dir)
        folder_names = [os.path.join(
            self.image_dir, nifti_file_name) for nifti_file_name in nifti_file_names if
            'N' or 'P' in nifti_file_name]
        test = ['Case_P005.nii.gz', 'Case_P008.nii.gz', 'Case_P039.nii.gz', 'Case_P057.nii.gz', 'Case_P061.nii.gz',
                'Case_P071.nii.gz', 'Case_P072.nii.gz', 'Case_P077.nii.gz', 'Case_P078.nii.gz', 'Case_P082.nii.gz']
        folder_names_filtered = [folder_name for folder_name in folder_names if
                                 os.path.basename(folder_name) not in test]
        return folder_names_filtered

    def __len__(self):
        return len(self.image_paths)

    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        img = tio.ScalarImage(image_path)

        image_name = image_path.split('/')[-1]
        mask_path = os.path.join(self.label_dir, image_name)
        mask = tio.LabelMap(mask_path)  

        img = self.preprocessing(img)
        mask = self.preprocessing(mask)

        img_affine = img.affine
        img = img.data.permute(0, -1, 1, 2)
        mask = mask.data.permute(0, -1, 1, 2)

        if (mask == 3).sum().item() == 0:
            hist = torch.zeros((16), dtype=torch.float)
        else:
            hist = torch.histc(img[mask == 3], bins=16, min=-1, max=1) / (mask == 3).sum()

        return {
            'GT': img,
            'GT_name': image_name,
            'gt_keep_mask': mask,
            'hist': hist,
            'affine': img_affine
        }



class EMIDECInDataset_Normal(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.image_paths = self.get_image_files()


    def get_image_files(self):
        nifti_file_names = os.listdir(self.image_dir)
        folder_names = [os.path.join(
            self.image_dir, nifti_file_name) for nifti_file_name in nifti_file_names if
            'P' in nifti_file_name]
        test = ['Case_P005.nii.gz', 'Case_P008.nii.gz', 'Case_P039.nii.gz', 'Case_P057.nii.gz', 'Case_P061.nii.gz',
                'Case_P071.nii.gz', 'Case_P072.nii.gz', 'Case_P077.nii.gz', 'Case_P078.nii.gz', 'Case_P082.nii.gz']
        folder_names_filtered = [folder_name for folder_name in folder_names if
                                 os.path.basename(folder_name) not in test]
        return folder_names_filtered

    def __len__(self):
        return len(self.image_paths)

    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        img = tio.ScalarImage(image_path)

        image_name = image_path.split('/')[-1]
        mask_path = os.path.join(self.label_dir, image_name)
        mask = tio.LabelMap(mask_path)

        img = self.preprocessing(img)
        mask = self.preprocessing(mask)

        img_affine = img.affine
        img = img.data.permute(0, -1, 1, 2)
        mask = mask.data.permute(0, -1, 1, 2)

        if (mask == 3).sum().item() == 0:
            hist = torch.zeros((16), dtype=torch.float)
        else:
            hist = torch.histc(img[mask == 3], bins=16, min=-1, max=1) / (mask == 3).sum()

        return {
            'GT': img,
            'GT_name': image_name,
            'gt_keep_mask': mask,
            'hist': hist,
            'affine': img_affine
        }




from torch.utils.data import DataLoader
from dataset import LIDCDataset, LIDCInDataset
from dataset import EMIDECDataset, EMIDECDataset_Normal, EMIDECInDataset, EMIDECInDataset_Normal


def get_inference_dataloader(dataset_root_dir, test_txt_dir,batch_size=1, drop_last=False, data_type=''):
    if data_type == 'lidc':
        train_dataset = LIDCInDataset(root_dir=dataset_root_dir, test_txt_dir=test_txt_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'emidec':
        train_dataset = EMIDECInDataset(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'emidec_normal':
        train_dataset = EMIDECInDataset_Normal(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    return loader

def get_inference_dataloader_norm(cfg):
    if cfg.dataset.data_type == 'emidec_diff_seg':
        inf_dataset = EMIDECInDataset(root_dir=cfg.dataset.root_dir_norm)
    return inf_dataset

def get_inference_dataloader_norm_valid(cfg):
    if cfg.dataset.data_type == 'emidec_diff_seg':
        inf_dataset = EMIDECInDataset(root_dir=cfg.dataset.root_dir_valid)
    return inf_dataset

def get_inference_dataloader_path(cfg):
    if cfg.dataset.data_type == 'emidec_diff_seg':
        inf_dataset = EMIDECInDataset_Normal(root_dir=cfg.dataset.root_dir_path)
    return inf_dataset



def get_train_dataset(cfg):
    if cfg.dataset.data_type == 'lidc':
        train_dataset = LIDCDataset(root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        sampler = None
    elif cfg.dataset.data_type == 'emidec':
        train_dataset = EMIDECDataset(root_dir=cfg.dataset.root_dir)
        sampler = None
    elif cfg.dataset.data_type == 'emidec_normal':
        train_dataset = EMIDECDataset_Normal(root_dir=cfg.dataset.root_dir)
        sampler = None
    return train_dataset, sampler

def get_train_dataset_norm(cfg):
    if cfg.dataset.data_type == 'emidec_diff_seg':
        train_dataset = EMIDECDataset_Normal(root_dir=cfg.dataset.root_dir_norm)
        sampler = None
    return train_dataset, sampler

def get_train_dataset_path(cfg):
    if cfg.dataset.data_type == 'emidec_diff_seg':
        train_dataset = EMIDECDataset(root_dir=cfg.dataset.root_dir_path)
        sampler = None
    return train_dataset, sampler
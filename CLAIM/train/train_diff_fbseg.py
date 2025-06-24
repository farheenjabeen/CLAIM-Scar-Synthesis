import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm.diffusion_fbseg import Unet3D, Trainer, GaussianDiffusion_Nolatent
import hydra
from omegaconf import DictConfig
from get_dataset.get_dataset import get_train_dataset_path, get_train_dataset_norm, get_inference_dataloader_path, get_inference_dataloader_norm
import torch
from ddpm.unet import UNet
import torch.nn as nn
from ddpm.model.base_segmentation_model import *
#--------------------------------------------------------------------------------
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.run.run_training import *
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List
import json
import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
#from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
#from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
#from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
#from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from ddpm.model.utils import LR_Scheduler
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    data_type = cfg.dataset.data_type.lower()
    if data_type not in ['emidec_diff_seg']:
        raise ValueError("Wrong data type")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_dim=cfg.model.cond_dim,
        )

    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        )
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    model = nn.DataParallel(model)

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        device=device,
        data_type=data_type
    ).to(device)

    plans_file = open(cfg.model.seg_nnUNet_plan_file, "r")
    data = json.loads(plans_file.read())
    architecture_class_name = data['configurations']['2d']['architecture']['network_class_name']
    arch_init_kwargs = data['configurations']['2d']['architecture']['arch_kwargs']
    arch_init_kwargs_req_import = data['configurations']['2d']['architecture']['_kw_requires_import']
    plans_file.close()
    num_input_channels = cfg.model.diffusion_num_channels
    enable_deep_supervision = True
    num_output_channels = cfg.model.seg_num_classes
    segmentor = get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision)
    segmentor = segmentor.to(device)
    # Load pretrained weights
    checkpoint = torch.load(cfg.model.seg_pretrained_weights, map_location=device, weights_only=False)
    new_state_dict = {}
    for k, value in checkpoint['network_weights'].items():
        key = k
        if key not in segmentor.state_dict().keys() and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    segmentor.load_state_dict(new_state_dict)
    print('................... pretrained weights loaded ...................')

    norm_train_dataset, *_ = get_train_dataset_norm(cfg)
    path_train_dataset, *_ = get_train_dataset_path(cfg)

    norm_inf_dataset = get_inference_dataloader_norm(cfg)
    path_inf_dataset = get_inference_dataloader_path(cfg)

    trainer = Trainer(
        diffusion,
        segmentor,
        cfg=cfg,
        norm_dataset=norm_train_dataset,
        path_dataset=path_train_dataset,
        inf_norm_dataset=norm_inf_dataset,
        inf_path_dataset=path_inf_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        device=device,
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    trainer.train()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------


if __name__ == '__main__':
    run()




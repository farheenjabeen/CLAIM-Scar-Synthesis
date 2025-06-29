import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm.diffusion_seg import Unet3D, Trainer, GaussianDiffusion_Nolatent
import hydra
from omegaconf import DictConfig
from get_dataset.get_dataset import get_train_dataset_path, get_train_dataset_norm, get_inference_dataloader_path, get_inference_dataloader_norm
import torch
from ddpm.unet import UNet
import torch.nn as nn
from ddpm.model.base_segmentation_model import *
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

    segmentor = SegmentationModel(
        network_type=cfg.model.seg_network_type,
        num_classes=cfg.model.seg_num_classes,
        in_channels=cfg.model.diffusion_num_channels,
        optimizer_name=cfg.model.seg_optimizer_name,
        decoder_dropout=False,
        use_gpu=True,
        lr=0.0001
    ).to(device)

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

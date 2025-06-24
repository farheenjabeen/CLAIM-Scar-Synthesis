import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import DataLoader
from collections import defaultdict
import os
import sys
import json
import io
import blobfile as bf
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
#------------------------------------------------------------------------------
import argparse
import os
import time
from os.path import exists, join
import gc
import socket
import numpy as np
import cv2
from skimage.transform import resize
import torch
import torch as th
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from ddpm.model.model_utils import makeVariable
from ddpm.common_utils.basic_operations import intensity_norm_fn, construct_input
from ddpm.common_utils.metrics import print_metric
from ddpm.get_adv_augmentor import get_default_augmentor
from ddpm.model.model_utils import get_scheduler, ExponentialMovingAverage
from ddpm.model.loss import *
#------------------------------------------------------------------------------
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
import gc
import torchvision
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('LeFusion/LeFusion_Model_FBSeg/EMIDEC/runs/LeFusion_Model_FBSeg')
#------------------------------------------------------------------------------

def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)

def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

def perturb_tensor(tensor, mean=0.0, std=1.0, bili=0.1):
    perturbation = th.normal(mean, std, size=tensor.size())
    perturbation -= perturbation.mean()
    max_perturbation = tensor.abs() * bili
    perturbation = perturbation / perturbation.abs().max() * max_perturbation
    perturbed_tensor = tensor + perturbation
    return perturbed_tensor

def _check_times(times, t_0, t_T):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)

def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):
    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1
    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1
    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1
    t = t_T
    ts = []
    while t >= 1:
        t = t-1
        ts.append(t)
        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)
        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)
        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1
        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1
    ts.append(-1)
    _check_times(ts, -1, t_T)

    return ts


def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8
    ):
        super().__init__()
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ): 
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        prob_focus_present=0.
    ):
        if cond is None:                    
            cond = torch.zeros((1, 16))
            cond[0, -1] = 1.0
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x = self.init_conv(x)
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = cond.to(device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)

            t = torch.cat((t, cond), dim=-1)
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion_Nolatent(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=2,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False, 
        dynamic_thres_percentile=0.9,
        device=None,
        data_type=''
    ):
        super().__init__()
        self.data_type = data_type
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.vqgan = None
        self.device=device
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type


        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))


        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))


        self.text_use_bert_cls = text_use_bert_cls


        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        if isinstance(self.denoise_fn, torch.nn.DataParallel):
            noise = self.denoise_fn.module.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        else:
            noise = self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=noise)
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)
        return img


    @torch.inference_mode()
    def p_sample_loop_repaint(self,
                              shape,
                              noise=None,
                              model_kwargs=None,
                              progress=True,
                              conf=None,
                              device=None,
                              cond=None,
                              cond_scale=1.,
                              ):
        final = None
        for sample in self.p_sample_loop_repaint_progressive(
                shape,
                noise=noise,
                model_kwargs=model_kwargs,
                progress=progress,
                conf=conf,
                cond_scale=cond_scale,
                device=device,
                cond=cond,
        ):
            final = sample
        return final

    def p_sample_repaint(
            self,
            x,
            t,
            cond=None,
            cond_scale=1.,
            clip_denoised=True,
            conf=None,
            model_kwargs=None,
    ):
        b, *_= x.shape
        gt_keep_mask = model_kwargs.get('gt_keep_mask')
        if gt_keep_mask is None:
            gt_keep_mask = conf.get_inpa_mask(x)
        gt = model_kwargs['gt']  
        if self.data_type == 'lidc':
            mask = (gt_keep_mask == 1).float()
        elif self.data_type == 'emidec':
            mask = (gt_keep_mask == 3).float()
        elif self.data_type == 'emidec_diff_seg':
            mask = (gt_keep_mask == 3).float()
        mask = mask.eq(0) 
        alpha_cumprod = _extract_into_tensor(
            self.alphas_cumprod, t, x.shape)
        if conf.inpa_inj_sched_prev_cumnoise:
            weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
        else:
            gt_weight = torch.sqrt(alpha_cumprod)
            gt_part = gt_weight * gt
            noise_weight = torch.sqrt((1 - alpha_cumprod))
            noise_part = noise_weight * torch.randn_like(x)
            weighed_gt = gt_part + noise_part
        x = (mask * (weighed_gt)+(~mask) * (x))
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise 


    def p_sample_loop_repaint_progressive(
        self,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None,
        cond=None,
        cond_scale=None
    ):
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = torch.randn(*shape, device=device)
        self.gt_noises = None  
        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)
        if conf.schedule_jump_params:
            times = get_schedule_jump(**conf.schedule_jump_params)
            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)
            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = t_last
                if t_cur < t_last: 
                    with (torch.no_grad()):
                        b = shape[0]
                        out = self.p_sample_repaint(image_after_step, torch.full((b,), t_last_t, device=device, dtype=torch.long), cond_scale=cond_scale,
                                                    model_kwargs=model_kwargs,
                                                    conf=conf,
                                                    cond=cond                 )
                        image_after_step = out
                        sample_idxs[t_cur] += 1
                        yield out
                else:
                    t_shift = conf.get('inpa_inj_time_shift', 1)
                    image_after_step = self.undo(
                        image_after_step,
                        t=t_last_t+t_shift)
    def undo(self, img_after_model, t):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)
        img_in_est = torch.sqrt(1 - beta) * img_out + \
                    torch.sqrt(beta) * torch.randn_like(img_out)
        return img_in_est


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, mask, cond=None, noise=None, **kwargs):
        device = x_start.device
        x_start = x_start.to(device=device, dtype=torch.float32)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)
        x_recon = self.denoise_fn(**dict(x=x_noisy, time=t, cond=cond, **kwargs))
        if self.data_type == 'lidc' :
            noise = noise * mask
            x_recon = x_recon * mask
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, x_recon)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        elif self.data_type == 'emidec':
            mask = (mask == 3).float()
            noise = noise * mask
            x_recon = x_recon * mask
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, x_recon)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        elif self.data_type == 'emidec_diff_seg':
            mask = (mask == 3).float()
            noise = noise * mask
            x_recon = x_recon * mask
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, x_recon)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        return loss

    def forward(self, x, mask, *args, **kwargs):
        if isinstance(x, tuple):
            x, h = x
        else:
            h = None
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels,
                    f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long().to(self.device)
        cond = h
        return self.p_losses(**dict(x_start=x, t=t, mask=mask, cond=cond, *args, **kwargs))


class GaussianDiffusion_Nolatent_Normal(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=2,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,
            dynamic_thres_percentile=0.9,
            device=None,
            data_type=''
    ):
        super().__init__()
        self.data_type = data_type
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.vqgan = None
        self.device = device
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.text_use_bert_cls = text_use_bert_cls

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        if isinstance(self.denoise_fn, torch.nn.DataParallel):
            noise = self.denoise_fn.module.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        else:
            noise = self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=noise)
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)
        return img

    @torch.inference_mode()
    def p_sample_loop_repaint(self,
                              shape,
                              noise=None,
                              model_kwargs=None,
                              progress=True,
                              conf=None,
                              device=None,
                              cond=None,
                              cond_scale=1.,
                              ):
        final = None
        for sample in self.p_sample_loop_repaint_progressive(
                shape,
                noise=noise,
                model_kwargs=model_kwargs,
                progress=progress,
                conf=conf,
                cond_scale=cond_scale,
                device=device,
                cond=cond,
        ):
            final = sample
        return final

    def p_sample_repaint(
            self,
            x,
            t,
            cond=None,
            cond_scale=1.,
            clip_denoised=True,
            conf=None,
            model_kwargs=None,
    ):
        b, *_ = x.shape
        gt_keep_mask = model_kwargs.get('gt_keep_mask')
        if gt_keep_mask is None:
            gt_keep_mask = conf.get_inpa_mask(x)
        gt = model_kwargs['gt']
        if self.data_type == 'lidc':
            mask = (gt_keep_mask == 1).float()
        elif self.data_type == 'emidec_normal':
            mask = (gt_keep_mask == 3).float()
        mask = mask.eq(0)
        alpha_cumprod = _extract_into_tensor(
            self.alphas_cumprod, t, x.shape)
        if conf.inpa_inj_sched_prev_cumnoise:
            weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
        else:
            gt_weight = torch.sqrt(alpha_cumprod)
            gt_part = gt_weight * gt
            noise_weight = torch.sqrt((1 - alpha_cumprod))
            noise_part = noise_weight * torch.randn_like(x)
            weighed_gt = gt_part + noise_part
        x = (mask * (weighed_gt) + (~mask) * (x))
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop_repaint_progressive(
            self,
            shape,
            noise=None,
            model_kwargs=None,
            device=None,
            progress=False,
            conf=None,
            cond=None,
            cond_scale=None
    ):
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = torch.randn(*shape, device=device)
        self.gt_noises = None
        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)
        if conf.schedule_jump_params:
            times = get_schedule_jump(**conf.schedule_jump_params)
            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)
            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = t_last
                if t_cur < t_last:
                    with (torch.no_grad()):
                        b = shape[0]
                        out = self.p_sample_repaint(image_after_step,
                                                    torch.full((b,), t_last_t, device=device, dtype=torch.long),
                                                    cond_scale=cond_scale,
                                                    model_kwargs=model_kwargs,
                                                    conf=conf,
                                                    cond=cond)
                        image_after_step = out
                        sample_idxs[t_cur] += 1
                        yield out
                else:
                    t_shift = conf.get('inpa_inj_time_shift', 1)
                    image_after_step = self.undo(
                        image_after_step,
                        t=t_last_t + t_shift)

    def undo(self, img_after_model, t):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)
        img_in_est = torch.sqrt(1 - beta) * img_out + \
                     torch.sqrt(beta) * torch.randn_like(img_out)
        return img_in_est

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod,
                        t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, mask, cond=None, noise=None, **kwargs):
        device = x_start.device
        x_start = x_start.to(device=device, dtype=torch.float32)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)
        x_recon = self.denoise_fn(**dict(x=x_noisy, time=t, cond=cond, **kwargs))
        if self.data_type == 'lidc':
            noise = noise * mask
            x_recon = x_recon * mask
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, x_recon)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        elif self.data_type == 'emidec_normal':
            mask = (mask == 3).float()
            noise = noise * mask
            x_recon = x_recon * mask
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, x_recon)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        return loss

    def forward(self, x, mask, *args, **kwargs):
        if isinstance(x, tuple):
            x, h = x
        else:
            h = None
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels,
                    f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long().to(self.device)
        cond = h
        return self.p_losses(**dict(x_start=x, t=t, mask=mask, cond=cond, *args, **kwargs))



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        segmentation_model,
        cfg,
        norm_dataset=None,
        path_dataset=None,
        inf_norm_dataset=None,
        inf_path_dataset=None,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=50001,
        gradient_accumulate_every,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        seg_save_and_sample_every=100,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
        device=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.seg_model = segmentation_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.seg_save_and_sample_every = seg_save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.device = device

        self.cfg = cfg

        self.path_ds = path_dataset
        self.norm_ds = norm_dataset
        self.inf_norm_ds = inf_norm_dataset
        self.inf_path_ds = inf_path_dataset

        norm_dl = DataLoader(self.norm_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        path_dl = DataLoader(self.path_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        inf_norm_dl = DataLoader(self.inf_norm_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        inf_path_dl = DataLoader(self.inf_path_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

        self.norm_len_dataloader = len(norm_dl)
        self.path_len_dataloader = len(path_dl)
        self.inf_norm_len_dataloader = len(inf_norm_dl)
        self.inf_path_len_dataloader = len(inf_path_dl)

        self.norm_dl = cycle(norm_dl)
        self.path_dl = cycle(path_dl)
        self.inf_norm_dl = cycle(inf_norm_dl)
        self.inf_path_dl = cycle(inf_path_dl)

        print(f'found {len(self.norm_ds)} normal images (for training) as .nii.gz files')
        print(f'found {len(self.path_ds)} pathological images (for training) as .nii.gz files')
        print(f'found {len(self.inf_norm_ds)} normal images (for inference) as .nii.gz files')
        print(f'found {len(self.inf_path_ds)} pathological images (for inference) as .nii.gz files')

        assert len(
            self.norm_ds) > 0, 'need to have at least 1 image to start training (although 1 is not great, try 100k)'
        assert len(
            self.path_ds) > 0, 'need to have at least 1 image to start training (although 1 is not great, try 100k)'
        assert len(
            self.inf_norm_ds) > 0, 'need to have at least 1 image to start inference (although 1 is not great, try 10k)'
        assert len(
            self.inf_path_ds) > 0, 'need to have at least 1 image to start inference (although 1 is not great, try 10k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        start_time = time.time()
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                # ---------------------------------------------------------------------------------------
                # Step 1:  Train Lefusion (input Pathological)
                path_data_frame = next(self.path_dl)
                path_data = path_data_frame['data'].to(self.device)
                path_mask = path_data_frame['label'].to(self.device)
                if 'hist' in path_data_frame:
                    path_hist = path_data_frame['hist'].cuda()
                else:
                    path_hist = None
                self.model.train()
                with autocast(enabled=self.amp):
                    diff_loss = self.model(**dict(
                        x=(path_data, path_hist),
                        mask=path_mask,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask))
                    self.scaler.scale(diff_loss / self.gradient_accumulate_every).backward()
            # ---------------------------------------------------------------------------------------
            if (self.step >= 5000 and self.step % 10 == 0):
                # Step 2: Infer from Lefusion (input Normal) and Compute Segmentation loss
                self.model.eval()
                current_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_dir, 'hist_clusters', 'emidec_clusters.json')
                with open(file_path, 'r') as f:
                    clusters = json.load(f)
                cluster_centers = clusters[0]['centers']
                #print("sampling...")
                with autocast(enabled=self.amp):
                    for type in range(self.cfg.model.types):
                        #print("type_of_cond:", type + 1)
                        in_hist = th.tensor(cluster_centers[type])
                        in_hist = perturb_tensor(tensor=in_hist)
                        in_hist = in_hist.unsqueeze(0)
                        batch = next(self.inf_norm_dl)
                        for k in batch.keys():
                            if isinstance(batch[k], th.Tensor):
                                batch[k] = batch[k].to(self.device)
                        model_kwargs = {}
                        model_kwargs["gt"] = batch['GT']
                        gt_keep_mask = batch.get('gt_keep_mask')
                        if gt_keep_mask is not None:
                            model_kwargs['gt_keep_mask'] = gt_keep_mask
                        batch_size = model_kwargs["gt"].shape[0]
                        sample_fn = self.model.p_sample_loop_repaint
                        output = sample_fn(
                            shape=(batch_size, self.cfg.model.diffusion_num_channels, self.cfg.model.diffusion_depth_size, self.cfg.model.diffusion_img_size, self.cfg.model.diffusion_img_size),
                            model_kwargs=model_kwargs,
                            device=self.device,
                            progress=True,
                            conf=self.cfg,
                            cond=in_hist
                        )
                        gen_path_data = output.to(self.device)
                        gen_path_mask = batch['gt_keep_mask']
                        gen_path_hist = batch['hist']
                #print("sampling complete")
                # ---------------------------------------------------------------------------------------
                # Compute Segmentation loss
                data1 = np.reshape(gen_path_data.cpu().numpy(), (batch_size, self.cfg.model.diffusion_img_size, self.cfg.model.diffusion_img_size))
                mask1 = np.reshape(gen_path_mask.cpu().numpy(), (batch_size, self.cfg.model.diffusion_img_size, self.cfg.model.diffusion_img_size))
                data = torch.zeros((batch_size, 1, self.cfg.model.seg_image_height, self.cfg.model.seg_image_width)).to(self.device)
                mask = torch.zeros((batch_size, 1, self.cfg.model.seg_image_height, self.cfg.model.seg_image_width)).to(self.device)
                for b in range(batch_size):
                    data[b, 0, :, :] = torch.from_numpy(resize(data1[b, :, :], (self.cfg.model.seg_image_height, self.cfg.model.seg_image_width)))
                    mask[b, 0, :, :] = torch.from_numpy(cv2.resize(mask1[b, :, :], (self.cfg.model.seg_image_height, self.cfg.model.seg_image_width), interpolation=cv2.INTER_NEAREST))
                data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
                with torch.no_grad():
                    self.seg_model.eval()
                    self.seg_model.is_training = False
                    seg_outputs = self.seg_model(data)[0]
                    seg_outputs_soft = torch.softmax(seg_outputs, dim=1)
                    seg_outputs_pred = torch.argmax(seg_outputs, dim=1)
                    segmentation_loss = get_ce_dice_loss_scar(pred=seg_outputs_pred.unsqueeze(1), targets=mask, num_classes=2)
                    dice_score = dice_score_for_scar(output=seg_outputs_pred.cpu().numpy(), target=mask.squeeze(1).cpu().numpy())
                    writer.add_scalar("Train/dice_score", dice_score, self.step)
                ##################################################################################################
                # Visualize results
                real_norm_v_data = torchvision.utils.make_grid(batch['GT'].squeeze(1))
                real_norm_v_mask = torchvision.utils.make_grid(batch.get('gt_keep_mask').squeeze(1) == 3.0)
                writer.add_image('(input) Infer from Lefusion: Images', real_norm_v_data)
                writer.add_image('(input) Infer from Lefusion: Masks', real_norm_v_mask)
                gen_norm_v_data = torchvision.utils.make_grid(data)
                gen_norm_v_mask = torchvision.utils.make_grid(mask == 3.0)
                pred_norm_v_mask = torchvision.utils.make_grid((seg_outputs_pred == 3.0).unsqueeze(1))
                writer.add_image('(output) Infer from Lefusion: Images', gen_norm_v_data)
                writer.add_image('(output) Infer from Lefusion: Masks', gen_norm_v_mask)
                writer.add_image('(seg output) Infer from nnUNet: Preds', pred_norm_v_mask)
                ##################################################################################################
            # ----------------------------------------------------------------------------------------------------
            if (self.step < 5000):
                segmentation_loss = torch.tensor([0.0]).to(self.device)
            loss = diff_loss + 0.01 * (segmentation_loss)
            log = {'loss': loss.item()}
            writer.add_scalar("Train/diff_loss", diff_loss, self.step)
            writer.add_scalar("Train/seg_loss", segmentation_loss, self.step)
            writer.add_scalar("Train/loss", loss, self.step)
            # ---------------------------------------------------------------------------------------------------------
            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema_model.eval()
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                self.save(milestone)
            log_fn(log)
            print(f'Diff Loss {self.step}: {loss.item()} = {diff_loss.item()}, {segmentation_loss.item()}')
            if (self.step >= 5000 and self.step % 10 == 0):
                print("--- %s seconds ---" % (time.time() - start_time))
                print('--------------------------------------------------------------')
                start_time = time.time()
            self.step += 1
            # ---------------------------------------------------------------------------------------
        print('training completed')
        writer.flush()
        gc.collect()
        torch.cuda.empty_cache()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_ce_dice_loss_scar(pred, targets=None, num_classes=2):
    if not targets is None:
        cross_loss = softmax_weighted_loss_scar(pred, targets, num_cls=num_classes)
        dc_loss = dice_loss_scar(pred, targets, num_cls=num_classes)
        loss = dc_loss + cross_loss
    else:
        loss = 0.
    loss = loss
    return loss

def dice_score_for_scar(output, target):
    eps = 1e-8
    o3 = (output == 3)
    t3 = (target == 3)
    if (np.count_nonzero(t3) == 0):
        scar_dice = 0.0
    else:
        intersect3 = np.sum(2 * (o3 * t3), axis=(0, 1)) + eps
        denominator3 = np.sum(o3, axis=(0, 1)) + np.sum(t3, axis=(0, 1)) + eps
        scar_dice = np.average(intersect3 / denominator3)
    return scar_dice
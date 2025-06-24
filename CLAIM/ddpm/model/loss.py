import torch
import torch.nn as nn
import torch.nn.functional as F
cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=4, eps=1e-7):
    seg_volume = target.clone()
    lv_volume = (seg_volume == 1)  # LV
    myo_volume = (seg_volume == 2)  # MYO
    scar_volume = (seg_volume == 3)  # Scar
    bg_volume = (seg_volume == 0)
    seg_volume = [bg_volume, lv_volume, myo_volume, scar_volume]
    seg_volume = torch.stack((seg_volume), dim=1)
    target = seg_volume.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:] * target[:,i,:,:])
        l = torch.sum(output[:,i,:,:])
        r = torch.sum(target[:,i,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def softmax_weighted_loss(output, target, num_cls=4):
    seg_volume = target.clone()
    lv_volume = (seg_volume == 1)  # LV
    myo_volume = (seg_volume == 2)  # MYO
    scar_volume = (seg_volume == 3)  # Scar
    bg_volume = (seg_volume == 0)
    seg_volume = [bg_volume, lv_volume, myo_volume, scar_volume]
    seg_volume = torch.stack((seg_volume), dim=1).squeeze(2)
    target = seg_volume.float()
    B, _, H, W = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :]
        targeti = target[:, i, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1, 2)) * 1.0 / torch.sum(target, (1, 2, 3)))
        weighted = torch.reshape(weighted, (-1, 1, 1)).repeat(1, H, W)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss


def dice_loss_scar(output, target, num_cls=2, eps=1e-7):
    seg_volume = target.clone()
    scar_volume = (seg_volume == 3).squeeze(1)  # Scar
    bg_volume = (seg_volume == 0).squeeze(1)
    seg_volume = [bg_volume, scar_volume]
    seg_volume = torch.stack((seg_volume), dim=1)
    target = seg_volume.float()

    seg_volume = output.clone()
    scar_volume = (seg_volume == 3).squeeze(1)  # Scar
    bg_volume = (seg_volume == 0).squeeze(1)
    seg_volume = [bg_volume, scar_volume]
    seg_volume = torch.stack((seg_volume), dim=1)
    output1 = seg_volume.float()

    for i in range(num_cls):
        num = torch.sum(output1[:, i, :, :] * target[:, i, :, :])
        l = torch.sum(output1[:, i, :, :])
        r = torch.sum(target[:, i, :, :])
        if i == 0:
            dice = 2.0 * num / (l + r + eps)
        else:
            dice += 2.0 * num / (l + r + eps)
    return 1.0 - 1.0 * dice / num_cls


def softmax_weighted_loss_scar(output, target, num_cls=2):
    seg_volume = target.clone()
    scar_volume = (seg_volume == 3).squeeze(1)  # Scar
    bg_volume = (seg_volume == 0).squeeze(1)
    seg_volume = [bg_volume, scar_volume]
    seg_volume = torch.stack((seg_volume), dim=1)
    target = seg_volume.float()

    seg_volume = output.clone()
    scar_volume = (seg_volume == 3).squeeze(1)  # Scar
    bg_volume = (seg_volume == 0).squeeze(1)
    seg_volume = [bg_volume, scar_volume]
    seg_volume = torch.stack((seg_volume), dim=1)
    output1 = seg_volume.float()

    B, _, H, W = output1.size()
    for i in range(num_cls):
        outputi = output1[:, i, :, :]
        targeti = target[:, i, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1, 2)) * 1.0 / torch.sum(target, (1, 2, 3)))
        weighted = torch.reshape(weighted, (-1, 1, 1)).repeat(1, H, W)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

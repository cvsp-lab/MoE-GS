#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def m_psnr(img1, img2, mask):
    """
    img1, img2: torch.Tensor [B, C, H, W], normalized [0,1]
    """
    if mask.dim() == 2:          # [H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)        # [1,1,H,W]
    elif mask.dim() == 3:        # [B, H, W]
        mask = mask.unsqueeze(1)                      # [B,1,H,W]
    if mask.shape[1] == 1 and img1.shape[1] > 1:     # 채널 브로드캐스트
        mask_b = mask.repeat(1, img1.shape[1], 1, 1) # [B,C,H,W]
    else:
        mask_b = mask

    diff2 = (img1 - img2) ** 2           # [B, C, H, W]
    diff2_masked = diff2 * mask_b

    # 배치별 합/개수
    mse_sum = diff2_masked.reshape(img1.shape[0], -1).sum(1, keepdim=True)             # [B,1]
    num_ones = mask_b.reshape(img1.shape[0], -1).sum(1, keepdim=True).clamp_min(1e-8)  # [B,1]

    
    mse = mse_sum / num_ones                    # divide by total pixel count
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def pixelwise_l1_map(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    img1, img2: [B, C, H, W]
    Returns:
        l1_map: [B, H, W] - 채널 평균 절대 오차
    """
    diff = torch.abs(img1 - img2)
    l1_map = diff.mean(dim=1)  # 채널 평균 -> [B, H, W]
    return l1_map

def pixelwise_psnr_map(img1, img2, eps=1e-8):
    """
    입력: [C, H, W] 또는 [1, C, H, W]
    출력: [H, W] – 픽셀별 PSNR 유사도 맵
    """
    if img1.dim() == 4:
        img1 = img1[0]
    if img2.dim() == 4:
        img2 = img2[0]

    mse_map = (img1 - img2) ** 2  # [C, H, W]
    mse_map = mse_map.mean(dim=0)  # channel-wise 평균 → [H, W]
    psnr_map = 20 * torch.log10(1.0 / torch.sqrt(mse_map + eps))  # 유사도 맵

    return psnr_map
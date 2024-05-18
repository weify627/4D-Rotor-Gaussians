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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def pearson2(da1, da2):
    assert len(da1.shape) == 1 or len(da1.shape)==2
    if len(da1.shape) == 1:
        da1 = da1.reshape(1, -1)
        da2 = da2.reshape(1, -1)
    k, n = da1.shape

    da1_mean = da1.mean(-1, keepdim=True)
    da1_var = (da1 - da1_mean) ** 2
    da1_var = da1_var.sum(-1, keepdim=True) / (n-1)

    da2_mean = da2.mean(-1, keepdim=True)
    da2_var = (da2-da2_mean)**2
    da2_var = da2_var.sum(-1, keepdim=True) / (n-1)

    da1da2cov = (da1 - da1_mean)  * (da2-da2_mean)
    da1da2cov = da1da2cov.sum(-1, keepdim=True) / (n-1)

    re = da1da2cov / (torch.sqrt(da1_var + 1e-7) * torch.sqrt(da2_var + 1e-7))

    return re.reshape(-1,)

def windowed_pearson(img1, img2, window_size=11, size_average=True, pixelnum=-1):
    channel = img1.size(-3)
    assert channel == 1

    if img1.dim() == 4:
        b, _, h, w = img1.shape
    else:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        b, _, h, w = img1.shape


    halfwindow = (window_size - 1) // 2
    img1_pad = F.pad(img1, [halfwindow, halfwindow, halfwindow, halfwindow], mode='reflect')
    img2_pad = F.pad(img2, [halfwindow, halfwindow, halfwindow, halfwindow], mode='reflect')

    unfolderop = torch.nn.Unfold(window_size)
    img1_patch = unfolderop(img1_pad).reshape(b, -1, h, w)
    img2_patch = unfolderop(img2_pad).reshape(b, -1, h, w)

    img1_patch_bhwxc = img1_patch.permute(0, 2, 3, 1).reshape(b*h*w, -1)
    img2_patch_bhwxc = img2_patch.permute(0, 2, 3, 1).reshape(b*h*w, -1)

    if pixelnum > 0:
        n_pix = img1_patch_bhwxc.shape[0]
        n_pix_select = torch.randint(low=0, high=n_pix, size=(512,)).to(img1_patch_bhwxc.device)
        img1_patch_bhwxc = torch.index_select(img1_patch_bhwxc, dim=0, index=n_pix_select)
        img2_patch_bhwxc = torch.index_select(img2_patch_bhwxc, dim=0, index=n_pix_select)
    

    return pearson2(img1_patch_bhwxc, img2_patch_bhwxc)


def pearson(img1, img2, size_average=True, pixelnum=4096):
    channel = img1.size(-3)
    assert channel == 1

    if img1.dim() == 4:
        b, _, h, w = img1.shape
    else:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        b, _, h, w = img1.shape
    
    import numpy as np
    pixidx = np.arange(h*w).reshape(h, w)
    pixavialbleidx = [pixidx[d[0]] for d in depthmasknp]
    pixavialbleidx = [np.random.choice(d, size=(pixelnum,)) for d in range(b)]
    pixavialbleidx = [torch.tensor(d, dtype=torch.long) for d in pixavialbleidx]

    depthgt = [gt_depth[di].reshape(-1,)[d] for di, d in enumerate(pixavialbleidx)]
    depthpre = [outputs["depth"][di].reshape(-1,)[d] for di, d in enumerate(pixavialbleidx)]

    depthgt = torch.stack(depthgt, dim=0)
    depthpre = torch.stack(depthpre, dim=0)


    halfwindow = (window_size - 1) // 2
    img1_pad = F.pad(img1, [halfwindow, halfwindow, halfwindow, halfwindow], mode='reflect')
    img2_pad = F.pad(img2, [halfwindow, halfwindow, halfwindow, halfwindow], mode='reflect')

    unfolderop = torch.nn.Unfold(window_size)
    img1_patch = unfolderop(img1_pad).reshape(b, -1, h, w)
    img2_patch = unfolderop(img2_pad).reshape(b, -1, h, w)

    img1_patch_bhwxc = img1_patch.permute(0, 2, 3, 1).reshape(b*h*w, -1)
    img2_patch_bhwxc = img2_patch.permute(0, 2, 3, 1).reshape(b*h*w, -1)

    if pixelnum > 0:
        n_pix = img1_patch_bhwxc.shape[0]
        n_pix_select = torch.randint(low=0, high=n_pix, size=(512,)).to(img1_patch_bhwxc.device)
        img1_patch_bhwxc = torch.index_select(img1_patch_bhwxc, dim=0, index=n_pix_select)
        img2_patch_bhwxc = torch.index_select(img2_patch_bhwxc, dim=0, index=n_pix_select)
    

    return pearson2(img1_patch_bhwxc, img2_patch_bhwxc)
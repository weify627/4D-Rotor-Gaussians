import cv2
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import lpips
import math

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())
loss_fn = lpips.LPIPS(net='vgg').cuda()

def PSNR(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    
    return psnr

def compute_psnr(img1, img2):
    mse = img2mse(img1, img2)
    psnr = mse2psnr(mse)
    return psnr

def gaussian(w_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()

def create_window(w_size, channel=1):
    _1D_window = gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window

def compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channel = img1.size(1)
    window = create_window(window_size, channel)
    window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=1).mean()

def compute_lpips(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)

    return loss_fn(img1, img2)

def calculate_metrics(path):
    gtpath=os.path.join(path,'gt-rgb')
    renderpath=os.path.join(path,'rgb')
    images_gt=os.listdir(gtpath)
    images_render=os.listdir(renderpath)
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips_vgg = 0.0
    num_images = len(images_gt)

    for image_name in images_render:
        gt_image = cv2.imread(os.path.join(gtpath, image_name))
        render_image = cv2.imread(os.path.join(renderpath, image_name))
        gt_image = torch.Tensor(gt_image).cuda() / 255.0
        render_image = torch.Tensor(render_image).cuda() / 255.0
        psnr = compute_psnr(gt_image, render_image)
        total_psnr += psnr
        ssim = compute_ssim(gt_image, render_image)
        total_ssim += ssim
        lpips_vgg = compute_lpips(gt_image, render_image)
        total_lpips_vgg += lpips_vgg

        #print(f"Image: {image_name}, PSNR: {psnr}")
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_lpips_vgg = total_lpips_vgg / num_images
    print(f"Average PSNR: {avg_psnr.item()}, SSIM: {avg_ssim.item()}, LPIPS-VGG: {avg_lpips_vgg.item()}")
    return avg_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="", help="input path to the video")
    args = parser.parse_args()
    calculate_metrics(args.path)

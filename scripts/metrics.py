import cv2
import os
import argparse
import torch
import numpy as np


def PSNR(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    
    return psnr


def calculate_psnr(path):
    gtpath=os.path.join(path,'gt-rgb')
    renderpath=os.path.join(path,'rgb')
    images_gt=os.listdir(gtpath)
    images_render=os.listdir(renderpath)
    total_psnr = 0.0
    num_images = len(images_gt)

    for image_name in images_render:
        gt_image = cv2.imread(os.path.join(gtpath, image_name))
        render_image = cv2.imread(os.path.join(renderpath, image_name))
        psnr = PSNR(gt_image, render_image)
        total_psnr += psnr

        #print(f"Image: {image_name}, PSNR: {psnr}")
    avg_psnr = total_psnr / num_images
    print(f"Average PSNR: {avg_psnr}")
    return avg_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="", help="input path to the video")
    args = parser.parse_args()
    calculate_psnr(args.path)

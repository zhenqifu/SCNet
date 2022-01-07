from __future__ import print_function
import argparse
import os
from PIL import Image
import numpy as np
import cv2
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# settings
parser = argparse.ArgumentParser(description='Performance')
parser.add_argument('--input_dir', type=str, default='results')
parser.add_argument('--reference_dir', default='../../Dataset/UIE/UIEBD/test/label')

opt = parser.parse_args()
print(opt)


im_path = opt.input_dir
re_path = opt.reference_dir
avg_psnr = 0
avg_ssim = 0
avg_lpips = 0
n = 0
loss_fn = lpips.LPIPS(net='alex')
loss_fn.cuda()
for filename in os.listdir(re_path):
    print(im_path + '/' + filename)
    n = n + 1
    im1 = Image.open(im_path + '/' + filename)
    im2 = Image.open(re_path + '/' + filename)

    (h, w) = im2.size
    im1 = im1.resize((h, w))
    im1 = np.array(im1)
    im2 = np.array(im2)

    score_psnr = psnr(im1, im2)
    score_ssim = ssim(im1, im2, multichannel=True)

    ex_p0 = lpips.im2tensor(cv2.resize(lpips.load_image(im_path + '/' + filename), (h, w)))
    ex_ref = lpips.im2tensor(lpips.load_image(re_path + '/' + filename))
    ex_p0 = ex_p0.cuda()
    ex_ref = ex_ref.cuda()
    score_lpips = loss_fn.forward(ex_ref, ex_p0)

    avg_psnr += score_psnr
    avg_ssim += score_ssim
    avg_lpips += score_lpips

avg_psnr = avg_psnr / n
avg_ssim = avg_ssim / n
avg_lpips = avg_lpips / n
print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips.item()))



import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr_ssim(sr, hr):
    sr = torch.squeeze(sr, dim=0)
    hr = torch.squeeze(hr, dim=0)
    sr = (sr*255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
    hr = (hr*255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

    return psnr(sr, hr), ssim(sr, hr, multichannel=True)


def PONO(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    out = (x - mean) / std
    return out, mean, std


def MS(x, beta, gamma):
    return x * gamma + beta


class Whiten2d(Module):

    def __init__(self, num_features, t=5, eps=1e-5, affine=True):
        super(Whiten2d, self).__init__()
        self.T = t
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):

        N, C, H, W = x.size()

        # N x C x (H x W)
        in_data = x.view(N, C, -1)

        eye = in_data.data.new().resize_(C, C)
        eye = torch.nn.init.eye_(eye).view(1, C, C).expand(N, C, C)

        # calculate other statistics
        # N x C x 1
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        # N x C x C
        cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)
        # N  x c x 1
        mean = mean_in
        cov = cov_in + self.eps * eye

        # perform whitening using Newton's iteration
        Ng, c, _ = cov.size()
        P = torch.eye(c).to(cov).expand(Ng, c, c)
        # reciprocal of trace of covariance
        rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)
        # whiten matrix: the matrix inverse of covariance, i.e., cov^{-1/2}
        wm = P.mul_(rTr.sqrt())

        x_hat = torch.bmm(wm, in_data - mean)
        x_hat = x_hat.view(N, C, H, W)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, self.num_features, 1, 1) + \
                self.bias.view(1, self.num_features, 1, 1)

        return x_hat

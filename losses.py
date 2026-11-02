"""The original three-term fidelity loss; works on CPU and CUDA.

The historical helper is named SSIM, but computes contrast/structure (CS)
without the luminance factor. Keep that expression for checkpoint consistency.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def cs_loss(img1, img2, size=11, sigma=1.5):
    if img1.shape != img2.shape or img1.ndim != 4 or img1.shape[1] != 1:
        raise ValueError('CS expects equally shaped Bx1xHxW tensors')
    if min(img1.shape[-2:]) < size:
        raise ValueError('Image must be at least 11x11 for the original valid CS window')
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    x = torch.from_numpy(x[None, None])
    y = torch.from_numpy(y[None, None])
    window = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    window = (window / window.sum()).to(device=img1.device, dtype=img1.dtype)
    mu1, mu2 = F.conv2d(img1, window), F.conv2d(img2, window)
    variance1 = F.conv2d(img1 * img1, window) - mu1 * mu1
    variance2 = F.conv2d(img2 * img2, window) - mu2 * mu2
    covariance = F.conv2d(img1 * img2, window) - mu1 * mu2
    return 1 - ((2 * covariance + 0.03 ** 2) / (variance1 + variance2 + 0.03 ** 2)).mean()


class FidelityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def gradient(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        gx, gy = F.conv2d(x, self.sobel_x), F.conv2d(x, self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def components(self, fused, ir, vis):
        intensity = F.l1_loss(fused, torch.max(ir, vis))
        gradient = F.l1_loss(self.gradient(fused), torch.max(self.gradient(ir), self.gradient(vis)))
        cs_ir, cs_vis = cs_loss(fused, ir), cs_loss(fused, vis)
        return intensity, gradient, cs_ir, cs_vis

    def forward(self, fused, ir, vis):
        intensity, gradient, cs_ir, cs_vis = self.components(fused, ir, vis)
        return intensity + 2.0 * gradient + 0.25 * (cs_ir + cs_vis)

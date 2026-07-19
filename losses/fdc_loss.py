"""
Frequency Decomposition Consistency Loss (FDC Loss)

设计目的：
    显式监督融合输出在频域上的内容保真度和结构保真度。

    从信号处理角度，图像可分解为：
    - 低频分量：决定整体强度分布、亮度、色彩（内容信息）
    - 高频分量：决定边缘、纹理、细节（结构信息）

    FDC loss 对这两个频带分别施加约束：
    - 低频：强度保真 + SSIM 一致性 → 确保内容信息保留
    - 高频：梯度保真 + 边缘保持 → 确保结构信息保留

    当分解尺度 σ→0 时，FDC 退化为 pixel-level fusion constraint。
    但通过频域分解，可以分析和解释约束在不同频带上的作用。

    论文表述：
    "We propose a Frequency Decomposition Consistency (FDC) loss that
    explicitly supervises the frequency-band fidelity of the fused output.
    By decomposing the fused image into low-frequency (content) and
    high-frequency (structure) components, we apply band-specific constraints:
    content preservation (intensity + SSIM) on the low-frequency component
    and structural fidelity (gradient consistency + edge preservation) on
    the high-frequency component."

组件：
    - loss_freq_low:  低频内容保真（强度 + SSIM）
    - loss_freq_high: 高频结构保真（梯度 + 边缘）
    - loss_freq_recon: 多尺度频域重建一致性

使用方式：
    >>> fdc_criterion = FrequencyDecompositionConsistencyLoss()
    >>> loss, logs = fdc_criterion(outputs, ir, vis)
    >>> total_loss = rec_loss + lambda_fdc * loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 工具函数
# ============================================================================

def gaussian_kernel2d(kernel_size, sigma, channels, device, dtype):
    """生成 2D Gaussian kernel（depth-wise conv）。"""
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def gaussian_blur(x, sigma):
    """可微 Gaussian blur（depth-wise conv + reflect padding）。"""
    if sigma <= 0:
        return x.clone()
    channels = x.shape[1]
    kernel_size = int(2 * round(3 * sigma) + 1)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_kernel2d(kernel_size, sigma, channels, x.device, x.dtype)
    padding = kernel_size // 2
    x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(x_padded, kernel, groups=channels)


def low_high_decompose(x, sigma):
    """低频/高频分解。"""
    low = gaussian_blur(x, sigma)
    high = x - low
    return low, high


def sobel_gradient(x):
    """可微 Sobel 梯度幅值。"""
    if x.shape[1] > 1:
        x = x[:, :1]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(x_pad, sobel_x)
    gy = F.conv2d(x_pad, sobel_y)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


# ============================================================================
# FDC Loss 主类
# ============================================================================

class FrequencyDecompositionConsistencyLoss(nn.Module):
    """
    FDC Loss — 频域分解一致性损失

    核心设计：将融合输出分解为低频和高频，分别约束。

    论文中的数学推导：
    设 f 为融合图像，s_IR 和 s_VIS 为源图。
    定义频域分解算子 D_σ: f → (L_σ(f), H_σ(f))，其中
      L_σ(f) = G_σ * f   (Gaussian 低通)
      H_σ(f) = f - L_σ(f) (高通残差)

    FDC loss 定义为：
      L_FDC = L_content(L_σ(f), L_σ(s)) + L_structure(H_σ(f), H_σ(s))

    其中：
      L_content = L1 + (1 - SSIM)  (内容保真)
      L_structure = L1 + ReLU       (结构保真)

    当 σ→0 时，L_σ(f)→f, H_σ(f)→0，FDC 退化为 pixel-level constraint。
    """

    def __init__(
        self,
        scales=(1.0, 2.0, 4.0),
        # 低频内容保真（对齐 fusion loss 强度项 + SSIM）
        lambda_intensity=1.0,
        lambda_ssim=0.5,
        # 高频结构保真（对齐 fusion loss 梯度项）
        lambda_gradient=2.0,
        lambda_edge=1.0,
        # 多尺度重建
        lambda_recon=0.15,
        detach_targets=True,
    ):
        """
        Args:
            scales: 多尺度 sigma 值
            lambda_intensity: 低频强度保真权重
            lambda_ssim: 低频 SSIM 保真权重
            lambda_gradient: 高频梯度一致性权重
            lambda_edge: 高频边缘保持权重
            lambda_recon: 多尺度重建一致性权重
            detach_targets: 是否 detach 目标值
        """
        super().__init__()
        self.scales = scales
        self.lambda_intensity = lambda_intensity
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        self.lambda_edge = lambda_edge
        self.lambda_recon = lambda_recon
        self.detach_targets = detach_targets

        # SSIM 参数
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim(self, x, y, window_size=11):
        """可微分 SSIM（单通道）。"""
        if x.shape[1] > 1:
            x = x[:, :1]
        if y.shape[1] > 1:
            y = y[:, :1]

        channels = 1
        coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
        g = torch.exp(-0.5 * (coords / 1.5) ** 2)
        g = g / g.sum()
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        padding = window_size // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        y_pad = F.pad(y, (padding, padding, padding, padding), mode='reflect')

        mu_x = F.conv2d(x_pad, kernel, groups=channels)
        mu_y = F.conv2d(y_pad, kernel, groups=channels)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x_pad ** 2, kernel, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y_pad ** 2, kernel, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x_pad * y_pad, kernel, groups=channels) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))

        return ssim_map.mean()

    def _scale_weight(self, sigma):
        """小尺度权重更高。"""
        return 1.0 / sigma

    # ==================================================================
    # loss_freq_low: 低频内容保真
    # ==================================================================

    def _loss_freq_low(self, fused, ir, vis):
        """
        低频内容保真损失。

        数学公式：
          L_content = Σ_σ w_σ * [ L1(L_σ(f), L_σ(s)) + (1 - SSIM(L_σ(f), L_σ(s))) ]

        其中 s = max(s_IR, s_VIS)（取两个源图低频的最大值）。

        物理意义：确保融合图的低频分量（整体强度分布）与源图一致。
        """
        loss_intensity = torch.tensor(0.0, device=fused.device, dtype=fused.dtype)
        loss_ssim = torch.tensor(0.0, device=fused.device, dtype=fused.dtype)
        total_weight = 0.0

        for sigma in self.scales:
            sw = self._scale_weight(sigma)

            fus_low, _ = low_high_decompose(fused, sigma)
            ir_low, _ = low_high_decompose(ir, sigma)
            vis_low, _ = low_high_decompose(vis, sigma)

            target_low = torch.max(ir_low, vis_low).detach()

            loss_intensity = loss_intensity + sw * F.l1_loss(fus_low, target_low)
            loss_ssim = loss_ssim + sw * (1.0 - self._ssim(fus_low, target_low))
            total_weight += sw

        if total_weight > 0:
            loss_intensity = loss_intensity / total_weight
            loss_ssim = loss_ssim / total_weight

        loss_low = self.lambda_intensity * loss_intensity + self.lambda_ssim * loss_ssim
        return loss_low, {
            "loss_freq_low": loss_low,
            "loss_freq_low_intensity": loss_intensity,
            "loss_freq_low_ssim": loss_ssim,
        }

    # ==================================================================
    # loss_freq_high: 高频结构保真
    # ==================================================================

    def _loss_freq_high(self, fused, ir, vis):
        """
        高频结构保真损失。

        数学公式：
          L_structure = Σ_σ w_σ * [ L1(H_σ(f), H_σ(s)) + ReLU(H_σ(s) - H_σ(f)) ]

        其中 s = max(s_IR, s_VIS)。

        物理意义：确保融合图的高频分量（边缘、纹理）与源图一致。
        ReLU 项惩罚融合图丢失边缘信息。
        """
        loss_gradient = torch.tensor(0.0, device=fused.device, dtype=fused.dtype)
        loss_edge = torch.tensor(0.0, device=fused.device, dtype=fused.dtype)
        total_weight = 0.0

        for sigma in self.scales:
            sw = self._scale_weight(sigma)

            _, fus_high = low_high_decompose(fused, sigma)
            _, ir_high = low_high_decompose(ir, sigma)
            _, vis_high = low_high_decompose(vis, sigma)

            target_high = torch.max(ir_high, vis_high).detach()

            loss_gradient = loss_gradient + sw * F.l1_loss(fus_high, target_high)
            loss_edge = loss_edge + sw * F.relu(target_high - fus_high).mean()
            total_weight += sw

        if total_weight > 0:
            loss_gradient = loss_gradient / total_weight
            loss_edge = loss_edge / total_weight

        loss_high = self.lambda_gradient * loss_gradient + self.lambda_edge * loss_edge
        return loss_high, {
            "loss_freq_high": loss_high,
            "loss_freq_high_gradient": loss_gradient,
            "loss_freq_high_edge": loss_edge,
        }

    # ==================================================================
    # loss_freq_recon: 多尺度频域重建一致性
    # ==================================================================

    def _loss_freq_recon(self, outputs, ir, vis):
        """
        多尺度频域重建一致性。
        监督重建图像在多尺度低/高频上接近源图。
        """
        loss_recon = torch.tensor(0.0, device=ir.device, dtype=ir.dtype)
        total_weight = 0.0

        recon_pairs = []
        for key, source in [("ir_recon", ir), ("vis_recon", vis),
                            ("ir_dec", ir), ("vis_dec", vis)]:
            if key in outputs and outputs[key] is not None:
                recon_pairs.append((outputs[key], source))

        if not recon_pairs:
            return loss_recon, {"loss_freq_recon": loss_recon}

        for sigma in self.scales:
            sw = self._scale_weight(sigma)
            for recon, source in recon_pairs:
                src_low, src_high = low_high_decompose(source, sigma)
                rec_low, rec_high = low_high_decompose(recon, sigma)
                loss_recon = loss_recon + sw * F.l1_loss(rec_low, src_low)
                loss_recon = loss_recon + sw * 2.0 * F.l1_loss(rec_high, src_high)
                total_weight += sw * 3.0

        if total_weight > 0:
            loss_recon = loss_recon / total_weight

        return loss_recon, {"loss_freq_recon": loss_recon}

    # ==================================================================
    # forward
    # ==================================================================

    def forward(self, outputs, ir, vis):
        """
        计算 FDC Loss。

        Args:
            outputs: dict, 模型输出
            ir:  [B, 1, H, W]
            vis: [B, 1, H, W]

        Returns:
            loss: scalar tensor
            loss_dict: dict
        """
        fused = outputs.get("fused", None)
        if fused is None:
            zero = torch.tensor(0.0, device=ir.device, dtype=ir.dtype)
            return zero, {
                "loss_fdc": zero, "loss_freq_low": zero, "loss_freq_high": zero,
                "loss_freq_recon": zero,
            }

        ll, dict_l = self._loss_freq_low(fused, ir, vis)
        lh, dict_h = self._loss_freq_high(fused, ir, vis)
        lr, dict_r = self._loss_freq_recon(outputs, ir, vis)

        loss_fdc = ll + lh + self.lambda_recon * lr

        loss_dict = {
            "loss_fdc": loss_fdc,
            "loss_freq_low": dict_l["loss_freq_low"],
            "loss_freq_low_intensity": dict_l["loss_freq_low_intensity"],
            "loss_freq_low_ssim": dict_l["loss_freq_low_ssim"],
            "loss_freq_high": dict_h["loss_freq_high"],
            "loss_freq_high_gradient": dict_h["loss_freq_high_gradient"],
            "loss_freq_high_edge": dict_h["loss_freq_high_edge"],
            "loss_freq_recon": dict_r["loss_freq_recon"],
        }

        return loss_fdc, loss_dict


# ============================================================================
# 单元测试
# ============================================================================

def test_fdc_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 1, 128, 128

    outputs = {
        "fused": torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1),
        "ir_recon": torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1),
        "vis_recon": torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1),
        "ir_dec": torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1),
        "vis_dec": torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1),
    }

    ir = torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1)
    vis = torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(0, 1)

    criterion = FrequencyDecompositionConsistencyLoss(
        scales=(1.0, 2.0, 4.0),
        lambda_intensity=1.0,
        lambda_ssim=0.5,
        lambda_gradient=2.0,
        lambda_edge=1.0,
        lambda_recon=0.15,
    ).to(device)

    loss, loss_dict = criterion(outputs, ir, vis)

    print(f"FDC Loss:  {loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")

    loss.backward()
    print("\n✓ Forward/Backward 测试通过")

    # 测试缺少 fused
    outputs_no_fused = {"ir_recon": torch.randn(B, C, H, W, device=device, requires_grad=True)}
    loss2, _ = criterion(outputs_no_fused, ir, vis)
    print(f"✓ 缺少 fused 时正常运行: loss={loss2.item():.6f}")

    assert 0 <= loss.item() < 100, f"Loss 数值异常: {loss.item()}"
    print("✓ 数值范围正常")


if __name__ == "__main__":
    test_fdc_loss()

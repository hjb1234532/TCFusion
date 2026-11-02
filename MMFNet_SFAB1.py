"""
MMFNet with Optimized Frequency Domain Decomposition
基于热传导的频域分解 + 跨模态引导 + 多步迭代融合

核心模块:
1. TCFD (Thermal Conduction Feature Decomposition) - IR单模态频率分解
2. CMHD (Cross-Modal Heat Diffusion) - VIS跨模态引导频率分解
3. FSHR (Frequency-domain Stepwise Hierarchical Reconstruction) - 多步迭代特征融合

优化目标: 最大化 EN, SD, PC, FMI, QE, QABF, SSIM, VIF
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


# ============================================================================
# Basic Building Blocks
# ============================================================================

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True,
                 norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size,
                                             padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size,
                                    padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.up(x)


# ============================================================================
# 核心模块1: Laplacian算子 (热传导核心)
# ============================================================================

class LaplacianConv(nn.Module):
    """
    离散Laplacian算子: ∇²u = ∂²u/∂x² + ∂²u/∂y²
    使用4邻域核: [[0,1,0], [1,-4,1], [0,1,0]]
    用于检测空间二阶导数(曲率/边缘强度)
    """

    def __init__(self, channels):
        super(LaplacianConv, self).__init__()
        # Laplacian kernel (4-neighborhood)
        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]], dtype=torch.float32)

        # Expand to channels
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', laplacian_kernel)
        self.channels = channels

    def forward(self, x):
        """
        Input: x (B, C, H, W)
        Output: Laplacian(x) (B, C, H, W) - 二阶导数,表征空间变化率
        """
        return F.conv2d(x, self.weight, padding=1, groups=self.channels)


# ============================================================================
# 核心模块2: 自适应扩散系数生成器
# ============================================================================

class AlphaGenerator(nn.Module):
    """
    生成空间自适应的扩散系数 α(x,y,t)

    核心思想:
    - 边缘区域: α小 → 扩散慢 → 保留细节
    - 平滑区域: α大 → 扩散快 → 去噪平滑

    输出范围: [0, α_max], α_max=0.5 (可学习)
    """

    def __init__(self, channels, alpha_max=0.5):
        super(AlphaGenerator, self).__init__()
        self.alpha_max = nn.Parameter(torch.tensor(alpha_max))

        # 使用轻量级网络生成α
        self.generator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出 [0,1]
        )

    def forward(self, x):
        """
        Input: x (B, C, H, W)
        Output: α (B, C, H, W), 范围 [0, α_max]
        """
        alpha_normalized = self.generator(x)  # [0, 1]
        alpha = alpha_normalized * torch.abs(self.alpha_max)  # [0, α_max]
        return alpha


# ============================================================================
# 核心模块3: 热响应图生成器 (用于跨模态引导)
# ============================================================================

class ThermalResponseMap(nn.Module):
    """
    生成IR热响应图 M_ir(x,y)

    作用: 识别IR中的热目标区域(高温目标)
    输出: [0,1]的空间权重图,热目标区域值接近1

    用于CMHD中引导VIS特征的扩散过程
    """

    def __init__(self, channels):
        super(ThermalResponseMap, self).__init__()
        self.response_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()  # 输出热响应强度 [0,1]
        )

    def forward(self, ir_feat):
        """
        Input: IR特征 (B, C, H, W)
        Output: 热响应图 M_ir (B, C, H, W), 范围 [0,1]
        """
        return self.response_net(ir_feat)


# ============================================================================
# 核心模块4: FSHR (频域分层重建) - 多步迭代特征融合
# ============================================================================

class FSHR(nn.Module):
    """
    Frequency-domain Stepwise Hierarchical Reconstruction

    核心作用:
    1. 整合多步迭代特征 u0~u3,解决单步特征的取舍困境
    2. 通过可学习权重β0~β3自动平衡细节与平滑
    3. 过滤高频噪声,提纯特征质量

    融合公式: F_fused = β0·u0 + β1·u1 + β2·u2 + β3·u3
    其中 β0~β3 经Softmax归一化,权重和为1
    """

    def __init__(self, channels, num_steps=4):
        super(FSHR, self).__init__()
        self.num_steps = num_steps

        # 可学习融合权重 (初始化为均匀分布)
        init_weights = torch.ones(num_steps) / num_steps
        self.fusion_weights = nn.Parameter(init_weights)

        # 特征增强层(可选,提升融合质量)
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, iteration_features):
        """
        Input: iteration_features - list of [u0, u1, u2, u3], 每个 (B,C,H,W)
        Output: F_fused (B, C, H, W) - 融合后的特征
        """
        # Softmax归一化权重
        beta = F.softmax(self.fusion_weights, dim=0)

        # 加权融合
        fused = 0
        for i, feat in enumerate(iteration_features):
            fused = fused + beta[i] * feat

        # 特征增强(残差连接)
        fused = fused + self.enhance(fused)

        return fused


# ============================================================================
# 核心模块5: TCFD (热传导特征分解) - IR单模态频率分解
# ============================================================================

class TCFD(nn.Module):
    """
    Thermal Conduction Feature Decomposition

    核心原理: 基于热传导方程的迭代扩散
    热传导方程: ∂u/∂t = α·∇²u
    离散形式: u_{t+1} = u_t + α·∇²u_t·Δt

    输入: IR编码器特征
    输出:
        - IR低频分量 (u3, 平滑特征, 全局结构)
        - IR高频分量 (原始 - 低频, 边缘纹理细节)
        - 中间迭代特征 [u0, u1, u2, u3] (用于FSHR)
    """

    def __init__(self, channels, num_steps=3, residual_scale=0.1):
        super(TCFD, self).__init__()
        self.num_steps = num_steps
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))

        # Laplacian算子
        self.laplacian = LaplacianConv(channels)

        # 扩散系数生成器
        self.alpha_gen = AlphaGenerator(channels, alpha_max=0.5)

        # FSHR融合模块
        self.fshr = FSHR(channels, num_steps=num_steps + 1)

    def forward(self, ir_feat, return_tcfd_info=False):
        """
        Input: ir_feat (B, C, H, W) - IR编码器特征
        Output:
            - low_freq (B, C, H, W) - IR低频分量
            - high_freq (B, C, H, W) - IR高频分量
            - fused_feat (B, C, H, W) - FSHR融合特征
        """
        # Step 1: 初始化
        u = ir_feat  # u0 = 原始特征
        iteration_features = [u]  # 存储所有迭代步
        alpha_map = None  # 存储最后一步的alpha

        # Step 2: 多步热传导迭代
        for t in range(self.num_steps):
            # a. 计算Laplacian (空间二阶导数)
            laplacian = self.laplacian(u)  # ∇²u

            # b. 生成自适应扩散系数
            alpha = self.alpha_gen(u)  # α(x,y,t)
            alpha_map = alpha  # 保存最后一步的alpha

            # c. 热传导更新
            # u_{t+1} = u_t + α·∇²u_t·residual_scale
            u = u + alpha * laplacian * torch.abs(self.residual_scale)
            iteration_features.append(u)

        # Step 3: 频率分离
        low_freq = u  # u3 = 最终迭代(平滑)
        high_freq = ir_feat - low_freq  # 高频 = 原始 - 低频

        # Step 4: FSHR多步融合
        fused_feat = self.fshr(iteration_features)

        if return_tcfd_info:
            return low_freq, high_freq, fused_feat, {
                'u_steps': iteration_features,
                'alpha': alpha_map
            }
        return low_freq, high_freq, fused_feat


# ============================================================================
# 核心模块6: CMHD (跨模态热扩散) - VIS跨模态引导频率分解
# ============================================================================

class CMHD(nn.Module):
    """
    Cross-Modal Heat Diffusion

    核心创新: 用IR热响应图引导VIS特征的频率分解

    跨模态调制机制:
        α_t = α0_t · (1 - γ·M_ir)
        - M_ir: IR热响应图 (0~1, 热目标区域接近1)
        - γ: 调制强度系数 (默认0.8)
        - 效果: 热目标区域α↓ → 扩散↓ → 保留VIS细节

    输入: VIS编码器特征 + IR编码器特征(用于生成M_ir)
    输出:
        - VIS低频分量 (受IR热区域引导)
        - VIS高频分量 (热区域更锐利)
        - 融合特征 (FSHR输出)
    """

    def __init__(self, channels, num_steps=3, residual_scale=0.1, gamma=0.8):
        super(CMHD, self).__init__()
        self.num_steps = num_steps
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))
        self.gamma = nn.Parameter(torch.tensor(gamma))  # 跨模态调制系数

        # Laplacian算子
        self.laplacian = LaplacianConv(channels)

        # 扩散系数生成器
        self.alpha_gen = AlphaGenerator(channels, alpha_max=0.5)

        # IR热响应图生成器
        self.thermal_map_gen = ThermalResponseMap(channels)

        # FSHR融合模块
        self.fshr = FSHR(channels, num_steps=num_steps + 1)

    def forward(self, vis_feat, ir_feat, return_tcfd_info=False):
        """
        Input:
            - vis_feat (B, C, H, W) - VIS编码器特征
            - ir_feat (B, C, H, W) - IR编码器特征(用于生成热响应图)
        Output:
            - low_freq (B, C, H, W) - VIS低频分量
            - high_freq (B, C, H, W) - VIS高频分量
            - fused_feat (B, C, H, W) - FSHR融合特征
        """
        # Step 1: 生成IR热响应图
        M_ir = self.thermal_map_gen(ir_feat)  # (B, C, H, W), [0,1]

        # Step 2: 初始化
        u = vis_feat  # u0 = VIS原始特征
        iteration_features = [u]
        alpha_map = None

        # Step 3: 跨模态引导的热传导迭代
        for t in range(self.num_steps):
            # a. 计算Laplacian
            laplacian = self.laplacian(u)

            # b. 生成基础扩散系数
            alpha0 = self.alpha_gen(u)

            # c. 跨模态调制 (核心创新)
            # 热目标区域(M_ir大) → α减小 → 扩散减弱 → 保留VIS细节
            alpha = alpha0 * (1 - torch.abs(self.gamma) * M_ir)
            alpha_map = alpha

            # d. 热传导更新
            u = u + alpha * laplacian * torch.abs(self.residual_scale)
            iteration_features.append(u)

        # Step 4: 频率分离
        low_freq = u  # u3 = 最终迭代
        high_freq = vis_feat - low_freq

        # Step 5: FSHR多步融合
        fused_feat = self.fshr(iteration_features)

        if return_tcfd_info:
            return low_freq, high_freq, fused_feat, {
                'u_steps': iteration_features,
                'alpha': alpha_map,
                'M_ir': M_ir
            }
        return low_freq, high_freq, fused_feat


# ============================================================================
# 改进的跨模态融合模块 (HCGFusionBlock替代)
# ============================================================================

class EnhancedCrossModalFusion(nn.Module):
    """
    增强型跨模态融合

    融合策略:
    1. 低频: 加权融合 (结构互补)
    2. 高频: Max融合 (保留最强细节)
    3. FSHR特征: 自适应融合
    """

    def __init__(self, channels):
        super(EnhancedCrossModalFusion, self).__init__()

        # 低频融合权重生成
        self.low_freq_weight_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)  # 对2个模态归一化
        )

        # FSHR特征融合权重
        self.fshr_weight_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 最终整合层
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, ir_low, ir_high, ir_fused, vis_low, vis_high, vis_fused):
        """
        Input:
            - ir_low, ir_high, ir_fused: IR的低频/高频/FSHR特征
            - vis_low, vis_high, vis_fused: VIS的低频/高频/FSHR特征
        Output:
            - fused_output: 最终融合特征
        """
        # 1. 低频加权融合
        low_concat = torch.cat([ir_low, vis_low], dim=1)
        low_weights = self.low_freq_weight_gen(low_concat)  # (B, 2, H, W)
        fused_low = low_weights[:, 0:1] * ir_low + low_weights[:, 1:2] * vis_low

        # 2. 高频Max融合 (保留最强细节)
        fused_high = torch.max(ir_high, vis_high)

        # 3. FSHR特征自适应融合
        fshr_concat = torch.cat([ir_fused, vis_fused], dim=1)
        fshr_weights = self.fshr_weight_gen(fshr_concat)
        fused_fshr = fshr_weights[:, 0:1] * ir_fused + fshr_weights[:, 1:2] * vis_fused

        # 4. 三分量整合
        # 低频(结构) + 高频(细节) + FSHR(多步优化)
        fused = fused_low + fused_high + fused_fshr

        # 5. 最终卷积整合
        output = self.final_conv(fused) + fused  # 残差连接

        return output


# ============================================================================
# Encoder & Decoder (保持原有结构)
# ============================================================================

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=1):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, num_res=1):
        super(Encoder, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=1),
        ])
        self.down12 = DownSample(base_channel * 2, base_channel * 2)
        self.down23 = DownSample(base_channel * 4, base_channel * 4)

    def forward(self, x):
        en_outputs = list()
        en_1 = self.feat_extract[0](x)
        res1 = self.Encoder[0](en_1)
        en_outputs.append(res1)
        en_2 = self.feat_extract[1](res1)
        res2 = self.Encoder[1](en_2)
        x_12 = self.down12(res2)
        en_outputs.append(x_12)
        en_3 = self.feat_extract[2](x_12)
        res3 = self.Encoder[2](en_3)
        x_23 = self.down23(res3)
        en_outputs.append(x_23)
        return en_outputs


class DBlock(nn.Module):
    def __init__(self, channel, num_res=1):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Skip(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Skip, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, num_res=1, is_seg=False, num_classes=9):
        super(Decoder, self).__init__()
        base_channel = 32
        self.is_seg = is_seg
        self.num_classes = num_classes
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])
        self.up32 = UpSample(base_channel * 2, base_channel * 2)
        self.up21 = UpSample(base_channel, base_channel)
        self.Skips = nn.ModuleList([
            Skip(base_channel * 7, base_channel * 1),
            Skip(base_channel * 7, base_channel * 2)
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.classifier = nn.Conv2d(base_channel, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs_12 = F.interpolate(inputs[0], scale_factor=0.5)
        inputs_32 = F.interpolate(inputs[2], scale_factor=2)
        inputs_21 = F.interpolate(inputs[1], scale_factor=2)
        inputs_31 = F.interpolate(inputs_32, scale_factor=2)
        skip_2 = self.Skips[1](inputs_12, inputs[1], inputs_32)
        skip_1 = self.Skips[0](inputs[0], inputs_21, inputs_31)
        res1 = self.Decoder[0](inputs[2])
        de_3 = self.feat_extract[0](res1)
        x_32 = self.up32(de_3)
        cat_2 = torch.cat([x_32, skip_2], dim=1)
        x_2 = self.Convs[0](cat_2)
        res2 = self.Decoder[1](x_2)
        de_2 = self.feat_extract[1](res2)
        x_21 = self.up21(de_2)
        cat_1 = torch.cat([x_21, skip_1], dim=1)
        x_1 = self.Convs[1](cat_1)
        res3 = self.Decoder[2](x_1)
        if self.is_seg:
            seg_head = self.classifier(res3)
            return seg_head
        else:
            de_1 = self.feat_extract[2](res3)
            return de_1


# ============================================================================
# 模态解耦模块 (Feature Decoupling Module)
# ============================================================================

class SimpleFDM(nn.Module):
    """简化的特征解耦模块"""

    def __init__(self, channels):
        super(SimpleFDM, self).__init__()
        self.coef_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, fused_feat, modality_feat):
        coef = self.coef_conv(modality_feat)
        decoupled = fused_feat - modality_feat * coef
        return decoupled


class MultiScaleFDM(nn.Module):
    """多尺度模态解耦"""

    def __init__(self, channels_list=[32, 64, 128]):
        super(MultiScaleFDM, self).__init__()
        self.fdm_1 = SimpleFDM(channels_list[0])
        self.fdm_2 = SimpleFDM(channels_list[1])
        self.fdm_3 = SimpleFDM(channels_list[2])

    def forward(self, fused_feats, modality_feats):
        return [
            self.fdm_1(fused_feats[0], modality_feats[0]),
            self.fdm_2(fused_feats[1], modality_feats[1]),
            self.fdm_3(fused_feats[2], modality_feats[2])
        ]


# ============================================================================
# 主网络: MMFNet with Frequency Domain Optimization
# ============================================================================

class MMFNet_FrequencyOptimized(nn.Module):
    """
    MMFNet with Optimized Frequency Domain Decomposition

    核心架构:
    1. Encoder: 提取多尺度IR/VIS特征
    2. TCFD: IR单模态频率分解 (热传导)
    3. CMHD: VIS跨模态频率分解 (IR引导)
    4. Enhanced Fusion: 增强型跨模态融合
    5. FDM: 模态解耦
    6. Decoder: 重建 + 分割

    优化目标: EN↑ SD↑ PC↑ FMI↑ QE↑ QABF↑ SSIM↑ VIF↑
    """

    def __init__(self, num_classes=9, diffusion_steps=2):
        super(MMFNet_FrequencyOptimized, self).__init__()
        self.num_classes = num_classes
        T = diffusion_steps  # 统一控制
        # Encoders
        self.encoder_ir = Encoder()
        self.encoder_vis = Encoder()

        # Frequency Decomposition Modules (多尺度)
        self.tcfd_1 = TCFD(channels=32, num_steps=T)  # Scale 1
        self.tcfd_2 = TCFD(channels=64, num_steps=T)  # Scale 2
        self.tcfd_3 = TCFD(channels=128, num_steps=T)  # Scale 3

        self.cmhd_1 = CMHD(channels=32, num_steps=T)
        self.cmhd_2 = CMHD(channels=64, num_steps=T)
        self.cmhd_3 = CMHD(channels=128, num_steps=T)

        # Enhanced Cross-Modal Fusion
        self.fusion_1 = EnhancedCrossModalFusion(channels=32)
        self.fusion_2 = EnhancedCrossModalFusion(channels=64)
        self.fusion_3 = EnhancedCrossModalFusion(channels=128)

        # Feature Decoupling Modules
        self.fdm_ir = MultiScaleFDM([32, 64, 128])
        self.fdm_vis = MultiScaleFDM([32, 64, 128])

        # Decoders
        self.decoder = Decoder(is_seg=False)
        self.seg_head = Decoder(is_seg=True, num_classes=num_classes)

        # Segmentation refinement
        self.seg_refine_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.seg_refine_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.seg_refine_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, ir, vis, return_decomposition=False, return_tcfd_info=False):
        """
        Input:
            - ir: IR图像 (B, 1, H, W)
            - vis: VIS图像 (B, 1, H, W)
        Output (训练模式):
            - recon_ir: IR重建 (B, 1, H, W)
            - recon_vis: VIS重建 (B, 1, H, W)
            - recon_dec_ir: 解耦后IR重建 (B, 1, H, W)
            - recon_dec_vis: 解耦后VIS重建 (B, 1, H, W)
            - seg_res: 分割结果 (B, num_classes, H, W)
            - fused: 融合图像 (B, 1, H, W)
        """
        # 1. Encoding
        ir_feats = self.encoder_ir(ir)  # [32, 64, 128] channels
        vis_feats = self.encoder_vis(vis)

        # 2. Single-modal reconstruction
        recon_ir = self.decoder(ir_feats)
        recon_vis = self.decoder(vis_feats)

        # 3. Frequency Decomposition (Multi-scale)
        all_tcfd_info = {}

        # Scale 1 (32 channels)
        if return_tcfd_info:
            ir_low_1, ir_high_1, ir_fused_1, tcfd_1_info = self.tcfd_1(ir_feats[0], return_tcfd_info=True)
            vis_low_1, vis_high_1, vis_fused_1, cmhd_1_info = self.cmhd_1(vis_feats[0], ir_feats[0], return_tcfd_info=True)
            all_tcfd_info['scale_1'] = {
                'u_steps_ir': tcfd_1_info['u_steps'],
                'alpha_ir': tcfd_1_info['alpha'],
                'u_steps_vis': cmhd_1_info['u_steps'],
                'alpha_vis': cmhd_1_info['alpha'],
            }
        else:
            ir_low_1, ir_high_1, ir_fused_1 = self.tcfd_1(ir_feats[0])
            vis_low_1, vis_high_1, vis_fused_1 = self.cmhd_1(vis_feats[0], ir_feats[0])

        # Scale 2 (64 channels)
        if return_tcfd_info:
            ir_low_2, ir_high_2, ir_fused_2, tcfd_2_info = self.tcfd_2(ir_feats[1], return_tcfd_info=True)
            vis_low_2, vis_high_2, vis_fused_2, cmhd_2_info = self.cmhd_2(vis_feats[1], ir_feats[1], return_tcfd_info=True)
            all_tcfd_info['scale_2'] = {
                'u_steps_ir': tcfd_2_info['u_steps'],
                'alpha_ir': tcfd_2_info['alpha'],
                'u_steps_vis': cmhd_2_info['u_steps'],
                'alpha_vis': cmhd_2_info['alpha'],
            }
        else:
            ir_low_2, ir_high_2, ir_fused_2 = self.tcfd_2(ir_feats[1])
            vis_low_2, vis_high_2, vis_fused_2 = self.cmhd_2(vis_feats[1], ir_feats[1])

        # Scale 3 (128 channels)
        if return_tcfd_info:
            ir_low_3, ir_high_3, ir_fused_3, tcfd_3_info = self.tcfd_3(ir_feats[2], return_tcfd_info=True)
            vis_low_3, vis_high_3, vis_fused_3, cmhd_3_info = self.cmhd_3(vis_feats[2], ir_feats[2], return_tcfd_info=True)
            all_tcfd_info['scale_3'] = {
                'u_steps_ir': tcfd_3_info['u_steps'],
                'alpha_ir': tcfd_3_info['alpha'],
                'u_steps_vis': cmhd_3_info['u_steps'],
                'alpha_vis': cmhd_3_info['alpha'],
            }
        else:
            ir_low_3, ir_high_3, ir_fused_3 = self.tcfd_3(ir_feats[2])
            vis_low_3, vis_high_3, vis_fused_3 = self.cmhd_3(vis_feats[2], ir_feats[2])

        # 4. Enhanced Cross-Modal Fusion
        fused_1 = self.fusion_1(ir_low_1, ir_high_1, ir_fused_1,
                                vis_low_1, vis_high_1, vis_fused_1)
        fused_2 = self.fusion_2(ir_low_2, ir_high_2, ir_fused_2,
                                vis_low_2, vis_high_2, vis_fused_2)
        fused_3 = self.fusion_3(ir_low_3, ir_high_3, ir_fused_3,
                                vis_low_3, vis_high_3, vis_fused_3)

        fused_feats = [fused_1, fused_2, fused_3]

        # 5. Feature Decoupling (模态解耦)
        dec_ir_feats = self.fdm_ir(fused_feats, vis_feats)
        dec_vis_feats = self.fdm_vis(fused_feats, ir_feats)

        # 6. Decoupled Reconstruction
        recon_dec_ir = self.decoder(dec_ir_feats)
        recon_dec_vis = self.decoder(dec_vis_feats)

        # 7. Segmentation
        seg_feats = [
            F.relu(self.seg_refine_1(fused_1)),
            F.relu(self.seg_refine_2(fused_2)),
            F.relu(self.seg_refine_3(fused_3))
        ]
        seg_res = self.seg_head(seg_feats)

        # 8. Fusion output
        fused = self.decoder(fused_feats)

        if return_tcfd_info:
            return recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, seg_res, fused, all_tcfd_info

        if return_decomposition:
            return recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, seg_res, fused, {
                'ir_low': [ir_low_1, ir_low_2, ir_low_3],
                'ir_high': [ir_high_1, ir_high_2, ir_high_3],
                'vis_low': [vis_low_1, vis_low_2, vis_low_3],
                'vis_high': [vis_high_1, vis_high_2, vis_high_3]
            }

        return recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, seg_res, fused


# Alias
MMFNet = MMFNet_FrequencyOptimized

# ============================================================================
# Testing
# ============================================================================

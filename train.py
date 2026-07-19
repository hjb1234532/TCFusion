"""
train_sfab1.py
使用 FDC Loss (Frequency Decomposition Consistency) 的训练脚本

FDC Loss 通过频域分解将融合输出拆分为低频/高频分量，
分别对两个频带施加针对性约束：
- 低频：内容保真度（强度 + SSIM）
- 高频：结构保真度（梯度 + 边缘）

总损失：
    loss = loss_recon + loss_seg + lambda_fdc * loss_fdc
"""

import warnings

warnings.filterwarnings("ignore")

import os
import kornia
from config import Config
from torch.utils.tensorboard import SummaryWriter

opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import utils
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from utils.MEF_SSIM_loss import th_SSIM_LOSS, Y_Upper
from utils.dice import DiceLoss
import numpy as np

# 使用改进版模型
from models.TCFusion_net import MMFNet
from losses.fdc_loss import FrequencyDecompositionConsistencyLoss


# ============================================================================
# 数据加载
# ============================================================================

def get_training_data(train_dir, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'whu':
        from data.data_RGB_whu import DataLoaderTrain
        return DataLoaderTrain(train_dir)
    elif dataset_name == 'mfnet':
        from data.data_RGB_mfnet import get_training_data as get_mfnet_data
        return get_mfnet_data(train_dir)
    elif dataset_name == 'potsdam':
        from data.data_RGB_potsdam import get_training_data as get_potsdam_data
        return get_potsdam_data(train_dir)
    else:
        from data.data_RGB_mfnet import get_training_data as get_mfnet_data
        return get_mfnet_data(train_dir)


def rgb_to_ycbcr(img):
    return kornia.color.rgb_to_ycbcr(img)




# ============================================================================
# 主训练循环
# ============================================================================

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 0
    data = opt.Datasets.data
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, data, 'models_sfab2_improved')
    utils.mkdir(model_dir)
    train_dir = opt.TRAINING.TRAIN_DIR

    print("=" * 70)
    print("SFAB1 Training - FDC Loss (Frequency Decomposition Consistency)")
    print("=" * 70)
    print(f"Dataset: {data}")
    print(f"Model dir: {model_dir}")
    print("\n损失函数:")
    print("  1. Reconstruction Loss: L1 + SSIM (自重建 + 交叉重建)")
    print("  2. Segmentation Loss: Dice + CrossEntropy")
    print("  3. FDC Loss: 频域分解一致性 (低频: 强度+SSIM, 高频: 梯度+边缘)")
    print("=" * 70)

    # TensorBoard
    log_dir = os.path.join(opt.TRAINING.SAVE_DIR, data, 'tensorboard_sfab2_improved')
    utils.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # 模型
    model = MMFNet(opt.TRAINING.NUM_CLASSES)
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999))

    # 学习率调度
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs, eta_min=opt.OPTIM.LR_MIN
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )
    scheduler.step()

    # 恢复训练
    if opt.TRAINING.RESUME:
        path_chk = utils.get_last_path(model_dir, 'model_latest.pth')
        if os.path.exists(path_chk):
            utils.load_checkpoint(model, path_chk)
            start_epoch = utils.load_start_epoch(path_chk)
            utils.load_optim(optimizer, path_chk)
            for i in range(1, start_epoch):
                scheduler.step()
            print(f"Resumed from epoch {start_epoch}")

    # 损失函数
    l1_loss = nn.L1Loss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(smooth=0.05, ignore_index=255)

    # FDC Loss (频域分解一致性)
    lambda_fdc = 1.0
    fdc_criterion = FrequencyDecompositionConsistencyLoss(
        scales=(1.0, 2.0, 4.0),
        lambda_intensity=1.0,   # 低频强度保真
        lambda_ssim=0.5,        # 低频 SSIM
        lambda_gradient=2.0,    # 高频梯度一致性（权重最大）
        lambda_edge=1.0,        # 高频边缘保持
        lambda_recon=0.15,      # 多尺度重建一致性
        detach_targets=True,
    ).cuda()
    print(f"FDC Loss: lambda_fdc={lambda_fdc}")

    # 数据加载
    train_dataset = get_training_data(train_dir, data)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    print(f'\nEpochs: {start_epoch} -> {opt.OPTIM.NUM_EPOCHS}')
    print(f'Samples: {len(train_dataset)}, Batch: {opt.OPTIM.BATCH_SIZE}')
    print("=" * 70)

    global_step = start_epoch * len(train_loader)

    # 训练循环
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        stats = {
            'total': 0, 'recon': 0, 'seg': 0, 'fdc': 0,
            'freq_low': 0, 'freq_high': 0, 'freq_recon': 0,
        }
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()

            target_ir = batch[0].cuda()
            input_ir = batch[1].cuda()
            target_rgb = batch[2].cuda()
            input_rgb = batch[3].cuda()
            target_seg = batch[4].cuda()

            input_ycbcr = rgb_to_ycbcr(input_rgb)
            target = Y_Upper(target_ir[:, :1], input_ycbcr[:, :1], 1.7).cuda()

            # 前向传播
            res_ir, res_y, res_rec_ir, res_rec_y, res_seg, fus = model(
                input_ir[:, :1], input_ycbcr[:, :1]
            )

            # RGB重建
            res_ycbcr = torch.cat([torch.clamp(res_y, 0, 1), input_ycbcr[:, 1:]], dim=1)
            res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)

            # ============ 损失计算 ============

            # 1. 重建损失
            loss_recon = (
                    l1_loss(res_ir, target_ir[:, :1]) +
                    l1_loss(res_rgb, target_rgb) +
                    l1_loss(res_rec_ir, target_ir[:, :1]) +
                    th_SSIM_LOSS(target, res_ir) +
                    th_SSIM_LOSS(target, res_y)
            )

            # 2. 分割损失
            try:
                loss_seg = ce_loss(res_seg, target_seg) + dice_loss(res_seg, target_seg)
            except:
                loss_seg = torch.tensor(0.0, device=fus.device)

            # 3. FDC Loss (频域分解一致性)
            fdc_outputs = {
                "fused": fus,
                "ir_recon": res_rec_ir,
                "vis_recon": res_y,
                "ir_dec": res_rec_ir,
                "vis_dec": res_rec_y,
            }
            loss_fdc, fdc_logs = fdc_criterion(
                fdc_outputs,
                input_ir[:, :1],
                input_ycbcr[:, :1]
            )

            # 总损失
            loss = loss_recon + loss_seg + lambda_fdc * loss_fdc

            # 反向传播
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                stats['total'] += loss.item()
                stats['recon'] += loss_recon.item()
                stats['seg'] += loss_seg.item()
                stats['fdc'] += loss_fdc.item()
                stats['freq_low'] += fdc_logs['loss_freq_low'].item()
                stats['freq_high'] += fdc_logs['loss_freq_high'].item()
                stats['freq_recon'] += fdc_logs['loss_freq_recon'].item()
                n_batches += 1

            # 打印
            if n_batches > 0 and i % 100 == 0:
                print(f"\n  [Batch {i}] 损失值:")
                print(f"    重建: {loss_recon.item():.4f}")
                print(f"    分割: {loss_seg.item():.4f}")
                print(f"    FDC: {loss_fdc.item():.4f}")
                print(f"      低频: {fdc_logs['loss_freq_low'].item():.4f} "
                      f"(强度: {fdc_logs['loss_freq_low_intensity'].item():.4f}, "
                      f"SSIM: {fdc_logs['loss_freq_low_ssim'].item():.4f})")
                print(f"      高频: {fdc_logs['loss_freq_high'].item():.4f} "
                      f"(梯度: {fdc_logs['loss_freq_high_gradient'].item():.4f}, "
                      f"边缘: {fdc_logs['loss_freq_high_edge'].item():.4f})")
                print(f"      重建: {fdc_logs['loss_freq_recon'].item():.4f}")
                print(f"    总损失: {loss.item():.4f}")

            # TensorBoard
            if n_batches > 0 and i % 10 == 0:
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/FDC', loss_fdc.item(), global_step)
                writer.add_scalar('Loss/FDC_Low', fdc_logs['loss_freq_low'].item(), global_step)
                writer.add_scalar('Loss/FDC_High', fdc_logs['loss_freq_high'].item(), global_step)
                writer.add_scalar('Loss/FDC_Recon', fdc_logs['loss_freq_recon'].item(), global_step)

            global_step += 1

            if n_batches > 0:
                pbar.set_postfix({
                    'loss': f"{stats['total'] / n_batches:.4f}",
                    'fdc': f"{stats['fdc'] / n_batches:.4f}",
                    'low': f"{stats['freq_low'] / n_batches:.4f}",
                    'high': f"{stats['freq_high'] / n_batches:.4f}",
                })

        # Epoch结束
        scheduler.step()

        if n_batches > 0:
            n = n_batches
            print(f"\n{'=' * 70}")
            print(f"[Epoch {epoch}] 完成 | 耗时: {time.time() - epoch_start:.1f}s")
            print(f"  平均损失:")
            print(f"    总损失: {stats['total'] / n:.4f}")
            print(f"    重建: {stats['recon'] / n:.4f}")
            print(f"    分割: {stats['seg'] / n:.4f}")
            print(f"    FDC: {stats['fdc'] / n:.4f}")
            print(f"      低频: {stats['freq_low'] / n:.4f}")
            print(f"      高频: {stats['freq_high'] / n:.4f}")
            print(f"      重建: {stats['freq_recon'] / n:.4f}")
            print(f"{'=' * 70}")

        # 保存模型
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_dir, "model_latest.pth"))

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    writer.close()
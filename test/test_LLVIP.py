import kornia
import numpy as np
import os
import argparse
from tqdm import tqdm
import time
import cv2  # 新增：导入cv2，用于指标计算

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data.data_RGB_LLVIP import get_test_data
from models.TCFusion_net import MMFNet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def rgb_to_ycbcr(img):
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr

# ===================== 新增：指标计算核心函数 =====================
def calculate_entropy(img):
    """计算图像熵 (EN) - 衡量信息丰富度"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图计算
    # 计算256级灰度直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化到概率分布
    hist = hist[hist > 0]     # 排除概率为0的灰度级（避免log2(0)报错）
    entropy = -np.sum(hist * np.log2(hist))  # 熵计算公式
    return entropy

def calculate_std(img):
    """计算标准差 (SD) - 衡量图像对比度/像素分布离散度"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.std(img.astype(np.float64))  # 高精度计算避免误差

def calculate_average_gradient(img):
    """计算平均梯度 (AG) - 衡量图像细节丰富度"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float64)
    # Sobel算子计算x/y方向梯度
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值并求平均
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(gradient)

def quick_metrics(img):
    """一键计算所有核心指标，返回字典"""
    en = calculate_entropy(img)
    sd = calculate_std(img)
    ag = calculate_average_gradient(img)
    return {'EN': en, 'SD': sd, 'AG': ag}

parser = argparse.ArgumentParser(description='Image Fusion using FreeFusion')
parser.add_argument('--input_dir', default='./dataset/test/', type=str, help='Directory for results')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/mfnet/models_sfab2_improved/model_latest.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_classes', default=9, type=int)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if __name__ == '__main__':
    model = MMFNet(args.num_classes)

    utils.load_checkpoint(model, args.weights)
    print("===>Testing using weights: ", args.weights)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    # 新增尺寸对齐函数
    def align_size(tensor, align=32):
        h, w = tensor.shape[2], tensor.shape[3]
        new_h = ((h + align - 1) // align) * align
        new_w = ((w + align - 1) // align) * align
        pad_h = new_h - h if new_h > h else 0
        pad_w = new_w - w if new_w > w else 0
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        tensor = tensor[:, :, :new_h, :new_w]
        return tensor, (h, w)

    datasets = ['FMB']

    for dataset in datasets:
        img_dir_test = os.path.join(args.input_dir, dataset)
        test_dataset = get_test_data(img_dir_test)
        # 修改num_workers=0
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=True)
        result_dir = os.path.join(args.result_dir, dataset)
        utils.mkdir(result_dir)

        # ===================== 新增：初始化指标存储 =====================
        metrics = {'EN': [], 'SD': [], 'AG': []}  # 存储所有图像的指标值

        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader), 0):
                torch.cuda.ipc_collect()
                inp_ir = data_test[0].cuda()
                inp_rgb = data_test[1].cuda()
                filenames = data_test[2]

                input_ycbcr = rgb_to_ycbcr(inp_rgb)

                # 尺寸对齐
                inp_ir_align, (ori_h, ori_w) = align_size(inp_ir[:, :1, :, :])
                input_ycbcr_align, _ = align_size(input_ycbcr[:, :1, :, :])

                # 模型推理
                _, _, _, _, _, fus = model(inp_ir_align, input_ycbcr_align)

                # 裁剪回原始尺寸
                fus = fus[:, :, :ori_h, :ori_w]

                fus = torch.clamp(fus, 0, 1)
                fus_ycbcr = torch.cat(
                    (torch.clamp(fus, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
                    dim=1)
                fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)
                ones = torch.ones_like(fus)
                zeros = torch.zeros_like(fus)
                fus_rgb = torch.where(fus_rgb > ones, ones, fus_rgb)
                fus_rgb = torch.where(fus_rgb < zeros, zeros, fus_rgb)
                fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()  # B C H W->B H W C
                fus_rgb = (fus_rgb - np.min(fus_rgb)) / (
                        np.max(fus_rgb) - np.min(fus_rgb)
                )
                fus_rgb = np.uint8(255.0 * fus_rgb)

                for batch in range(len(fus)):
                    fus_img = fus_rgb[batch]
                    utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), fus_img)

                    # ===================== 新增：计算并存储指标 =====================
                    img_metrics = quick_metrics(fus_img)  # 计算当前图像的指标
                    for key in metrics:
                        metrics[key].append(img_metrics[key])

            # ===================== 新增：统计并打印指标结果 =====================
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset} - 融合图像指标统计")
            print(f"{'=' * 60}")
            print(f"{'Metric':<10} {'Average Value':<15}")
            print(f"{'-' * 60}")
            for key in metrics:
                avg_value = np.mean(metrics[key])  # 计算所有图像的指标平均值
                print(f"{key:<10} {avg_value:<15.4f}")
            print(f"{'=' * 60}\n")
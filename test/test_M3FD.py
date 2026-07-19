import kornia
import numpy as np
import os
import argparse
from tqdm import tqdm
import time
import multiprocessing

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data.data_RGB_M3FD import get_test_data
from models.TCFusion_net import MMFNet

# 屏蔽无关警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== 新增：模型复杂度分析工具 ====================
def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_shape=(1, 1, 256, 256)):
    """
    计算模型FLOPs
    需要安装: pip install thop
    如果thop不可用，会尝试使用ptflops
    """
    try:
        from thop import profile, clever_format
        # 创建两个输入（IR和RGB的Y通道）
        input_ir = torch.randn(input_shape).cuda()
        input_rgb = torch.randn(input_shape).cuda()

        flops, params = profile(model, inputs=(input_ir, input_rgb))
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        print("Warning: thop not installed. Trying ptflops...")
        try:
            from ptflops import get_model_complexity_info
            # ptflops需要单输入，这里仅作估算
            macs, params = get_model_complexity_info(
                model, (1, 256, 256),
                as_strings=True,
                print_per_layer_stat=False
            )
            return macs, params
        except ImportError:
            print("Warning: Neither thop nor ptflops installed. Skipping FLOPs calculation.")
            print("Install with: pip install thop  or  pip install ptflops")
            return "N/A", "N/A"
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return "N/A", "N/A"


def rgb_to_ycbcr(img):
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr


# 核心：将所有执行逻辑封装到main函数
def main():
    parser = argparse.ArgumentParser(description='Image Fusion using FreeFusion')
    parser.add_argument('--input_dir', default='D:\hjb\FreeFusion/FreeFusion/dataset/test/', type=str,
                        help='Directory for results')
    parser.add_argument('--result_dir', default='./results', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./checkpoints/mfnet/models_sfab2_improved/model_latest.pth', type=str,
                        help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--calc_flops', action='store_true', help='Calculate FLOPs (requires thop/ptflops)')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model = MMFNet(args.num_classes)
    utils.load_checkpoint(model, args.weights)
    print("===>Testing using weights: ", args.weights)
    model.cuda()

    # ==================== 模型复杂度统计 ====================
    print("\n" + "=" * 60)
    print("Model Complexity Analysis")
    print("=" * 60)

    # 1. 参数量统计
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. FLOPs统计（可选）
    if args.calc_flops:
        print("\nCalculating FLOPs (this may take a moment)...")
        flops, params = calculate_flops(model)
        print(f"FLOPs: {flops}")
        print(f"Params (from FLOPs calc): {params}")

    print("=" * 60 + "\n")

    model = nn.DataParallel(model)
    model.eval()

    datasets = ['M3FD']

    # 用于统计推理时间
    inference_times = []

    for dataset in datasets:
        img_dir_test = os.path.join(args.input_dir, dataset)
        test_dataset = get_test_data(img_dir_test)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False
        )
        result_dir = os.path.join(args.result_dir, dataset)
        utils.mkdir(result_dir)

        print(f"\nProcessing dataset: {dataset}")
        print(f"Total images: {len(test_dataset)}")

        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader, desc="Inference"), 0):

                inp_ir = data_test[0].cuda()
                inp_rgb = data_test[1].cuda()
                filenames = data_test[2]

                input_ycbcr = rgb_to_ycbcr(inp_rgb)

                # ==================== 单张推理时间统计 ====================
                start_time = time.time()

                _, _, _, _, _, fus = model(inp_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])

                torch.cuda.synchronize()  # 仅在计时结束时同步一次
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)
                # ========================================================

                fus = torch.clamp(fus, 0, 1)
                fus_ycbcr = torch.cat(
                    (torch.clamp(fus, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
                    dim=1)
                fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)
                ones = torch.ones_like(fus)
                zeros = torch.zeros_like(fus)
                fus_rgb = torch.where(fus_rgb > ones, ones, fus_rgb)
                fus_rgb = torch.where(fus_rgb < zeros, zeros, fus_rgb)
                fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()
                fus_rgb = (fus_rgb - np.min(fus_rgb)) / (
                        np.max(fus_rgb) - np.min(fus_rgb)
                )
                fus_rgb = np.uint8(255.0 * fus_rgb)
                for batch in range(len(fus)):
                    fus_img = fus_rgb[batch]
                    utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), fus_img)

    # ==================== 推理时间统计结果 ====================
    if inference_times:
        print("\n" + "=" * 60)
        print("Inference Time Statistics")
        print("=" * 60)
        print(f"Total images processed: {len(inference_times)}")
        print(f"Average inference time: {np.mean(inference_times):.2f} ms")
        print(f"Median inference time: {np.median(inference_times):.2f} ms")
        print(f"Min inference time: {np.min(inference_times):.2f} ms")
        print(f"Max inference time: {np.max(inference_times):.2f} ms")
        print(f"Std deviation: {np.std(inference_times):.2f} ms")

        # 排除第一张图片后的统计（去除warm-up影响）
        if len(inference_times) > 1:
            print(f"\nExcluding first image (warm-up):")
            print(f"Average inference time: {np.mean(inference_times[1:]):.2f} ms")
            print(f"Median inference time: {np.median(inference_times[1:]):.2f} ms")

        print("=" * 60)

        # 保存详细统计到文件
        stats_file = os.path.join(args.result_dir, 'inference_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Model Complexity and Inference Statistics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: MMFNet\n")
            f.write(f"Weights: {args.weights}\n")
            f.write(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)\n")
            if args.calc_flops:
                f.write(f"FLOPs: {flops}\n")
            f.write(f"\nTotal images: {len(inference_times)}\n")
            f.write(f"Average inference time: {np.mean(inference_times):.2f} ms\n")
            f.write(f"Median inference time: {np.median(inference_times):.2f} ms\n")
            f.write(f"Min inference time: {np.min(inference_times):.2f} ms\n")
            f.write(f"Max inference time: {np.max(inference_times):.2f} ms\n")
            f.write(f"Std deviation: {np.std(inference_times):.2f} ms\n")
            if len(inference_times) > 1:
                f.write(f"\nExcluding first image (warm-up):\n")
                f.write(f"Average inference time: {np.mean(inference_times[1:]):.2f} ms\n")

        print(f"\nStatistics saved to: {stats_file}")


# Windows多进程必须的保护逻辑
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
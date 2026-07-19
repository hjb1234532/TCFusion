import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia

import utils
from data.data_RGB_whu import get_test_data
from models.TCFusion_net import MMFNet


def rgb_to_ycbcr(img):
    return kornia.color.rgb_to_ycbcr(img)


def main():
    parser = argparse.ArgumentParser(description='Image Fusion using FreeFusion')
    parser.add_argument('--input_dir', default='./dataset/test/', type=str)
    parser.add_argument('--result_dir', default='./results/', type=str)
    parser.add_argument('--weights', default='./checkpoints/whu/models_sfab_sky_fixed/model_latest.pth', type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--num_classes', default=8, type=int)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model = MMFNet(args.num_classes)
    utils.load_checkpoint(model, args.weights)
    print("===>Testing using weights:", args.weights)

    model = model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    datasets = ['whu']
    for dataset in datasets:
        img_dir_test = os.path.join(args.input_dir, dataset)
        test_dataset = get_test_data(img_dir_test)

        # ✅ Windows 下先确保跑通：num_workers=0
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,          # 先设 0
            drop_last=False,
            pin_memory=True
        )

        result_dir = os.path.join(args.result_dir, dataset)
        utils.mkdir(result_dir)

        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader), 0):
                inp_ir = data_test[0].cuda(non_blocking=True)
                inp_rgb = data_test[1].cuda(non_blocking=True)
                filenames = data_test[2]

                input_hsv = kornia.color.rgb_to_hsv(inp_rgb)
                _, _, _, _, _, fus = model(inp_ir[:, :1, :, :], input_hsv[:, 2:, :, :])

                fus_hsv = torch.cat(
                    (input_hsv[:, 0:1, :, :], input_hsv[:, 1:2, :, :], torch.clamp(fus, 0, 1)),
                    dim=1
                )
                fus_rgb = kornia.color.hsv_to_rgb(fus_hsv)

                fus_rgb = fus_rgb.clamp(0, 1)
                fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().numpy()  # B H W C
                fus_rgb = (fus_rgb - fus_rgb.min()) / (fus_rgb.max() - fus_rgb.min() + 1e-8)
                fus_rgb = np.uint8(255.0 * fus_rgb)

                for b in range(fus_rgb.shape[0]):
                    utils.save_img(os.path.join(result_dir, filenames[b] + '.png'), fus_rgb[b])


if __name__ == "__main__":
    # Windows 多进程/打包相关，建议加上
    import multiprocessing as mp
    mp.freeze_support()
    main()

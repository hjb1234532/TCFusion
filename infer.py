"""Fuse paired native-resolution images with a strict original T=2 checkpoint."""
import argparse
from pathlib import Path
import kornia
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from data import pair_paths, rgb_tensor
from runtime import build_model, device_for, fusion_only, load_checkpoint, setup, sha256, write_json


def legacy_postprocess(fused, ycbcr):
    fused = torch.clamp(fused, 0, 1)
    ycbcr = torch.cat((torch.clamp(fused, 0, 1), ycbcr[:, 1:2], ycbcr[:, 2:]), dim=1)
    rgb = kornia.color.ycbcr_to_rgb(ycbcr)
    rgb = torch.where(rgb > torch.ones_like(fused), torch.ones_like(fused), rgb)
    rgb = torch.where(rgb < torch.zeros_like(fused), torch.zeros_like(fused), rgb)
    array = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
    denominator = np.max(array) - np.min(array)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError('Original RGB min-max is undefined for constant/nonfinite output')
    array = (array - np.min(array)) / denominator
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite exported output')
    return np.uint8(255.0 * array)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ir-dir', required=True)
    parser.add_argument('--vis-dir', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    setup(args.seed, args.threads)
    device = device_for(args.device)
    pairs = pair_paths(args.ir_dir, args.vis_dir)
    model = build_model()
    checkpoint = load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for name, ir_path, vis_path in pairs:
            ir = rgb_tensor(ir_path).unsqueeze(0).to(device)[:, :1]
            rgb = rgb_tensor(vis_path).unsqueeze(0).to(device)
            if ir.shape[-2:] != rgb.shape[-2:]:
                raise ValueError('Unaligned source dimensions: ' + name)
            if min(ir.shape[-2:]) < 4:
                raise ValueError('Inference images must be at least 4x4')
            ycbcr = kornia.color.rgb_to_ycbcr(rgb)
            visible = ycbcr[:, :1]
            h, w = ir.shape[-2:]
            bottom, right = (-h) % 4, (-w) % 4
            if bottom or right:
                ir = F.pad(ir, (0, right, 0, bottom), mode='replicate')
                visible = F.pad(visible, (0, right, 0, bottom), mode='replicate')
            fused = fusion_only(model, ir, visible)[..., :h, :w]
            output = args.output / (name + '.png')
            Image.fromarray(legacy_postprocess(fused, ycbcr)).save(output)
            records.append({'id': name, 'native_hw': [h, w], 'pad_bottom_right': [bottom, right],
                            'ir_sha256': sha256(ir_path), 'vis_sha256': sha256(vis_path), 'output_sha256': sha256(output)})
            print('%d/%d %s' % (len(records), len(pairs), output), flush=True)
    write_json(args.output / 'inference.json', {'T': 2, 'checkpoint_sha256': sha256(args.checkpoint),
               'checkpoint_epoch': checkpoint.get('epoch'), 'strict_load': True, 'count': len(records), 'images': records,
               'export': 'native size; replicate right/bottom padding; original RGB min-max + uint8 truncation'})


if __name__ == '__main__':
    main()

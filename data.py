"""Explicit IR/VIS pairing by filename stem; no implicit resize or crop."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def image_map(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError('Duplicate image stem: ' + path.stem)
            result[path.stem] = path
    if not result:
        raise ValueError('No images in ' + str(directory))
    return result


def pair_paths(ir_dir, vis_dir):
    ir, vis = image_map(ir_dir), image_map(vis_dir)
    if set(ir) != set(vis):
        raise ValueError('IR/VIS filename stems differ: IR-only=%s, VIS-only=%s' %
                         (sorted(set(ir) - set(vis))[:10], sorted(set(vis) - set(ir))[:10]))
    return [(name, ir[name], vis[name]) for name in sorted(ir)]


def rgb_tensor(path):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div(255)


class PairedImages(Dataset):
    def __init__(self, ir_dir, vis_dir):
        self.pairs = pair_paths(ir_dir, vis_dir)
        shape = None
        for name, ir, vis in self.pairs:
            with Image.open(ir) as a, Image.open(vis) as b:
                if a.size != b.size:
                    raise ValueError('Unaligned image sizes: ' + name)
                if min(a.size) < 12 or any(n % 4 for n in a.size):
                    raise ValueError('Training images need H/W >=12 and divisible by four: ' + name)
                if shape is not None and a.size != shape:
                    raise ValueError('Training images must have a common size for batching')
                shape = a.size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name, ir, vis = self.pairs[index]
        # This is the original loader convention: IR first RGB channel, not RGB-to-Y.
        return rgb_tensor(ir)[:1], rgb_tensor(vis), name

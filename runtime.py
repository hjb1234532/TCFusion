"""T=2 construction, deterministic setup and strict checkpoint loading."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from MMFNet_SFAB1 import MMFNet


def setup(seed=1234, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device_for(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu to run on CPU')
    return device


def build_model():
    return MMFNet(num_classes=9, diffusion_steps=2)


def load_checkpoint(model, path):
    # Only load trusted checkpoints: torch 2.1 uses Python pickle.
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('diffusion_steps', 2) != 2:
        raise ValueError('This repository only runs T=2 checkpoints')
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state = {(key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def fusion_only(model, ir, visible):
    ir_features, vis_features = model.encoder_ir(ir), model.encoder_vis(visible)
    fused = []
    for i in range(1, 4):
        a = getattr(model, 'tcfd_%d' % i)(ir_features[i - 1])
        b = getattr(model, 'cmhd_%d' % i)(vis_features[i - 1], ir_features[i - 1])
        fused.append(getattr(model, 'fusion_%d' % i)(*a, *b))
    return model.decoder(fused)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)

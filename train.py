"""Train only original T=2 with the three-term fidelity objective."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import kornia
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import PairedImages
from losses import FidelityLoss
from runtime import build_model, device_for, fusion_only, load_checkpoint, setup, sha256, write_json

DEFAULT_CONFIG = Path(__file__).parent / 'configs/train_t2.json'


def learning_rate(epoch, cfg):
    warmup = min(cfg['warmup_epochs'], cfg['epochs'])
    if epoch <= warmup:
        return cfg['lr'] * epoch / warmup
    progress = (epoch - warmup) / max(1, cfg['epochs'] - warmup)
    return cfg['min_lr'] + (cfg['lr'] - cfg['min_lr']) * (1 + math.cos(math.pi * progress)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    for name in ['ir-dir', 'vis-dir', 'output', 'device']:
        parser.add_argument('--' + name)
    for name in ['epochs', 'batch-size', 'seed', 'workers', 'threads', 'save-every']:
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--resume', type=Path, help='Explicit resume of a checkpoint created by this trainer')
    parser.add_argument('--stop-after-epoch', type=int, help='Save and stop at this epoch, keeping the planned LR schedule')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if cfg['diffusion_steps'] != 2:
        raise ValueError('This release supports only T=2')
    if min(cfg['epochs'], cfg['batch_size'], cfg['threads'], cfg['save_every'], cfg['warmup_epochs']) < 1 or cfg['workers'] < 0:
        raise ValueError('Invalid training count')
    if not 0 <= cfg['min_lr'] <= cfg['lr']:
        raise ValueError('Require 0 <= min_lr <= lr')
    setup(cfg['seed'], cfg['threads'])
    device = device_for(cfg['device'])
    output = Path(cfg['output'])
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty; choose another --output or explicitly --resume')
    dataset = PairedImages(cfg['ir_dir'], cfg['vis_dir'])
    if len(dataset) < cfg['batch_size']:
        raise ValueError('Dataset smaller than batch size; lower --batch-size')
    manifest = [{'id': n, 'ir_sha256': sha256(ir), 'vis_sha256': sha256(vis)} for n, ir, vis in dataset.pairs]
    output.mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
    criterion = FidelityLoss().to(device)
    generator = torch.Generator().manual_seed(cfg['seed'])
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume)
        if checkpoint.get('schema') != 'tcfusion-t2-train-v1':
            raise ValueError('Legacy weights support inference; exact resume requires this trainer\'s optimizer/RNG metadata')
        if checkpoint['data_manifest'] != manifest:
            raise ValueError('Training data changed since the checkpoint')
        for key in ['epochs', 'batch_size', 'seed', 'lr', 'min_lr', 'warmup_epochs', 'diffusion_steps']:
            if checkpoint['config'][key] != cfg[key]:
                raise ValueError('Resume configuration differs: ' + key)
        optimizer.load_state_dict(checkpoint['optimizer'])
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        generator.set_state(checkpoint['rng']['loader'])
        if device.type == 'cuda' and checkpoint['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        start_epoch = checkpoint['epoch'] + 1
    write_json(output / 'config.json', cfg)
    write_json(output / 'data_manifest.json', manifest)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, generator=generator,
                        num_workers=cfg['workers'], drop_last=True, pin_memory=device.type == 'cuda')
    print(json.dumps({'T': 2, 'device': str(device), 'samples': len(dataset), 'batches': len(loader),
                      'initialization': 'explicit resume' if args.resume else 'random from scratch', 'config': cfg}), flush=True)
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        rate = learning_rate(epoch, cfg)
        for group in optimizer.param_groups:
            group['lr'] = rate
        totals = np.zeros(4, dtype=np.float64)
        started = time.time()
        for batch, (ir, rgb, _) in enumerate(loader, 1):
            ir, rgb = ir.to(device), rgb.to(device)
            visible = kornia.color.rgb_to_ycbcr(rgb)[:, :1]
            optimizer.zero_grad(set_to_none=True)
            fused = fusion_only(model, ir, visible)
            components = criterion.components(fused, ir, visible)
            loss = components[0] + 2 * components[1] + .25 * (components[2] + components[3])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            totals += np.array([v.item() for v in components])
            if batch == 1 or batch % 20 == 0 or batch == len(loader):
                print(json.dumps({'epoch': epoch, 'batch': batch, 'batches': len(loader), 'loss': loss.item(), 'lr': rate}), flush=True)
        means = totals / len(loader)
        summary = {'epoch': epoch, 'loss': float(means[0] + 2 * means[1] + .25 * (means[2] + means[3])),
                   'intensity': means[0], 'gradient_unweighted': means[1], 'cs_ir_unweighted': means[2],
                   'cs_vis_unweighted': means[3], 'lr': rate, 'seconds': time.time() - started}
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
               'loader': generator.get_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        checkpoint = {'schema': 'tcfusion-t2-train-v1', 'epoch': epoch, 'diffusion_steps': 2,
                      'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': cfg,
                      'data_manifest': manifest, 'rng': rng, 'summary': summary}
        temporary = output / 'model_latest.pth.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(output / 'model_latest.pth')
        if epoch % cfg['save_every'] == 0 or epoch == cfg['epochs']:
            torch.save(checkpoint, output / ('model_epoch_%03d.pth' % epoch))
        with (output / 'epochs.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(summary) + '\n')
        write_json(output / 'status.json', {**summary, 'status': 'completed' if epoch == cfg['epochs'] else 'training'})
        print(json.dumps(summary), flush=True)
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch and epoch < cfg['epochs']:
            write_json(output / 'status.json', {**summary, 'status': 'stopped_at_requested_epoch'})
            break


if __name__ == '__main__':
    main()

# TCFusion

Thermal Conduction Guided Fusion Network — an infrared-visible image fusion system with simultaneous semantic segmentation.

TCFusion models fusion as a heat conduction physical process: feature decomposition uses iterative thermal diffusion equations, and cross-modal fusion uses physically-inspired diffusion modules.

## Architecture

### Core Modules

| Module | Full Name | Function |
|--------|-----------|----------|
| **TCFD** | Thermal Conduction Feature Decomposition | IR single-modal frequency decomposition via iterative Laplacian diffusion |
| **CMHD** | Cross-Modal Heat Diffusion | VIS cross-modal frequency decomposition guided by IR thermal response |
| **FSHR** | Frequency-domain Stepwise Hierarchical Reconstruction | Multi-step iterative feature fusion with learned weights |
| **FDC Loss** | Frequency Decomposition Consistency Loss | Explicit frequency-band supervision on fused output |

### Forward Pass

The model takes paired IR + visible images (Y channel from YCbCr) and produces:
1. `recon_ir` — self-reconstruction of IR
2. `recon_vis` — self-reconstruction of visible
3. `recon_dec_ir` — cross-reconstruction (fused minus VIS features)
4. `recon_dec_vis` — cross-reconstruction (fused minus IR features)
5. `seg_res` — segmentation map (if dataset has labels)
6. `fus` — final fused image

### Loss Components

- **Reconstruction**: L1 + SSIM (self and cross)
- **Segmentation**: CrossEntropy + Dice
- **FDC Loss**: Frequency Decomposition Consistency (low-freq: intensity + SSIM, high-freq: gradient + edge)

## Quick Start

### Training

```bash
python train.py --config training.yml
```

Override config from command line:
```bash
python train.py --config training.yml --dataset potsdam --NUM_CLASSES 6
```

### Testing

```bash
python test/test_mfnet.py
python test/test_M3FD.py
python test/test_LLVIP.py
python test/test_whu.py
python test/test_potsdam.py
```

### Evaluation

```bash
python evaluate/batch_eval.py
```

## Project Structure

```
TCFusion/
├── train.py                    # Main training script
├── config.py                   # Configuration class (yacs-based)
├── training.yml                # Training hyperparameters
├── models/
│   └── TCFusion_net.py        # Network architecture (TCFD + CMHD + FSHR)
├── losses/
│   └── fdc_loss.py             # Frequency Decomposition Consistency Loss
├── utils/
│   ├── model_utils.py          # Checkpoint save/load
│   ├── MEF_SSIM_loss.py        # SSIM loss + FusionLoss
│   ├── dice.py                 # Dice loss for segmentation
│   ├── functional.py           # Focal loss, soft dice, etc.
│   ├── dir_utils.py            # Directory utilities
│   └── image_utils.py          # PSNR, image saving
├── data/
│   ├── data_RGB_mfnet.py       # MFNet data loader wrapper
│   ├── dataset_RGB_mfnet.py    # MFNet dataset implementation
│   ├── data_RGB_M3FD.py        # M3FD data loader wrapper
│   ├── dataset_RGB_M3FD.py     # M3FD dataset implementation
│   ├── data_RGB_LLVIP.py       # LLVIP data loader wrapper
│   ├── dataset_RGB_LLVIP.py    # LLVIP dataset implementation
│   ├── data_RGB_whu.py         # WHU data loader wrapper
│   ├── dataset_RGB_whu.py      # WHU dataset implementation
│   ├── data_RGB_potsdam.py     # Potsdam data loader wrapper
│   └── dataset_RGB_potsdam.py  # Potsdam dataset implementation
├── test/
│   ├── test_mfnet.py           # Test on MFNet
│   ├── test_M3FD.py            # Test on M3FD
│   ├── test_LLVIP.py           # Test on LLVIP
│   ├── test_whu.py             # Test on WHU
│   └── test_potsdam.py         # Test on Potsdam
├── evaluate/
│   ├── batch_eval.py           # Batch evaluation
│   └── fusion_metrics.py       # Fusion quality metrics
├── checkpoints/                # Model checkpoints
├── dataset/                    # Training/test data
└── requirements.txt
```

## Datasets

| Dataset | NUM_CLASSES | Segmentation |
|---------|-------------|-------------|
| MFNet | 9 | Yes |
| Potsdam | 6 | Yes |
| WHU | 8 | Yes |
| M3FD | — | No |
| LLVIP | — | No |

### Data Layout

```
dataset/train/{dataset}/
  ir/input/     ir/target/     rgb/input/     rgb/target/     seg/
dataset/test/{dataset}/
  ir/           rgb/
```

## Color Space

All models operate on **single-channel Y (luminance)** from YCbCr color space. Input images are converted via `kornia.rgb_to_ycbcr()`, and the Y channel is extracted with `[:,0:1,:,:]`. Cb/Cr channels are preserved and recombined after fusion.

## Requirements

```
torch>=1.9.0
kornia
yacs
natsort
warmup-scheduler
tensorboard
tqdm
opencv-python
numpy
Pillow
```

## Acknowledgment

Built on top of [FreeFusion](https://github.com/lyu-yx/FreeFusion).

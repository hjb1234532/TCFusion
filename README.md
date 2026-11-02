# TCFusion · 原始 T=2

原始 TCFusion 两步模型的独立运行版本，包含网络、三项损失、从头训练、断点续训和图像融合入口。默认种子为 **1234**。仓库只提供源代码、配置和运行验证，不包含数据集、权重、融合图片或指标文件。

## 模型与损失

`MMFNet_SFAB1.py` 保留原始网络实现。IR 和 VIS 的三个编码尺度均执行 **两次扩散更新**，读取初始状态与两个中间状态，经原有 FSHR 和跨模态融合模块得到融合图像。未加入后来实验中的分解替代方案。

仅使用正文中的三个损失项：

```text
L = L1(If, max(IR, VIS_Y))
  + 2 × L1(g(If), max(g(IR), g(VIS_Y)))
  + 0.25 × [CS_loss(If, IR) + CS_loss(If, VIS_Y)]
```

`g` 为 Sobel 梯度幅值。`CS_loss` 是原实现的对比度／结构损失（11×11 Gaussian，sigma=1.5，valid 卷积），不包含完整 SSIM 的亮度因子。没有额外的分割、重建或中间步监督损失。

网络保留六输出接口及原有参数，便于严格加载历史权重。训练与推理入口只执行其中的融合路径；测试核对该路径与六输出接口的融合结果、训练梯度一致。

## 环境

已验证 Python 3.8、PyTorch 2.1.0、Kornia 0.7.3。推荐在独立环境安装，CUDA 版 PyTorch 须与机器驱动匹配；CPU 也可运行。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

测试会在临时目录生成小型图像，完成前向／梯度一致性验证、两轮训练、断点恢复一致性和非整倍数尺寸图像融合；无需下载数据或权重。

## 从头训练

准备已经配准、裁切好的训练图像，两个目录内文件的主文件名必须一一对应（扩展名可以不同）：

```text
dataset/train/mfnet/
├── ir/input/001.png
├── ir/input/002.png
├── rgb/input/001.png
└── rgb/input/002.png
```

原训练使用 320×320 图块。本入口不隐式缩放或裁切；所有训练图像尺寸须相同，宽高是 4 的倍数且至少 12 像素。无需标签、target 目录或预训练权重。

```bash
python train.py --config configs/train_t2.json
```

也可以显式指定目录：

```bash
python train.py --ir-dir /path/to/train/ir --vis-dir /path/to/train/vis \
  --output runs/t2_seed1234 --epochs 150 --batch-size 4 --seed 1234 --device cuda:0
```

默认随机初始化，Adam，学习率 2e-5，3 轮 warmup 后余弦下降至 1e-6。每轮保存 `model_latest.pth`、`epochs.jsonl` 与 `status.json`，每 10 轮另存阶段权重。非空输出目录默认拒绝覆盖。

续训须显式指定本入口生成的权重，并保持原配置、数据和计划总轮数：

```bash
python train.py --config configs/train_t2.json \
  --resume runs/t2_seed1234/model_latest.pth
```

断点保存模型、优化器、随机数与数据加载器状态。`--stop-after-epoch N` 可在第 N 轮后保存停止，不修改原计划学习率。训练文件哈希用于检测续训时的数据变更。

**训练入口说明：** 本版本整理了数据读取、设备选择和保存逻辑；`--epochs 150` 明确表示 150 次完整遍历，编号为 1–150。旧脚本从 0 到 150（含两端），实际为 151 次遍历，且学习率调度器调用时序不同。本入口不是旧训练日志的逐步重放；原始模型、损失公式和历史权重的推理路径保持一致。

## 图像融合

先训练得到权重，或自行提供原始 T=2 权重。本仓库不附带预训练权重。

```bash
python infer.py --checkpoint runs/t2_seed1234/model_latest.pth \
  --ir-dir /path/to/TNO/ir --vis-dir /path/to/TNO/vis \
  --output results/TNO --device cuda:0
```

M3FD、LLVIP、MFNet、TNO、HarvardMIF 均可使用同一入口，分别指定对应的两组图像目录与独立输出目录。输入须已配准、同尺寸，宽高至少 4 像素。`--device cpu --threads 2` 可用于 CPU 推理。

按原代码约定，IR 取 PIL RGB 的第一通道，VIS 转 YCbCr 后取 Y。非 4 整倍数的尺寸仅在右／下边缘复制填充，输出裁回原尺寸。导出沿用原有处理：限制融合 Y 范围，拼回 VIS 色度，转 RGB 并限制范围，再进行整张 RGB 的 min-max 归一化、uint8 截断保存 PNG。纯常数或非有限输出会报错。

支持原历史权重的 `state_dict`／`model`／纯参数字典，以及 `module.` 前缀；严格检查参数键与形状。历史权重缺少 T 元数据时须由使用者确认是原始 T=2。旧训练断点可用于推理；本入口的精确续训要求使用本入口保存的优化器与随机数元数据。

输出目录中的 `inference.json` 记录权重、输入及输出哈希、图像数量和补边尺寸。此仓库提供融合推理，不包含正式 MATLAB 指标工具；不要将其他近似实现的指标与论文结果混用。

## 文件

| 文件 | 用途 |
|---|---|
| `MMFNet_SFAB1.py` | 原始 T=2 网络及六输出接口 |
| `losses.py` | 正文三项损失，CPU/CUDA 通用 |
| `data.py` | 按文件名配对和尺寸检查 |
| `runtime.py` | T=2 构建、严格加载、融合路径 |
| `train.py` | 从头训练与显式续训 |
| `infer.py` | 原尺寸融合与原导出处理 |
| `configs/train_t2.json` | 默认训练配置 |
| `tests/test_release.py` | 独立运行验证 |

原始网络文件 SHA-256：`b8cf8da18014843380e4e5721fe071fe0c872ecee187306f1bfb2a8c4a87c96c`。

## 致谢

本项目基于 [FreeFusion](https://github.com/lyu-yx/FreeFusion) 开展研究，保留原仓库的来源说明。

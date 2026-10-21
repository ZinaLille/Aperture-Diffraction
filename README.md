# Aperture-Diffraction

光阑形状 → 衍射图样的正向预测。工程只保留三个脚本，分别负责 **按 NF 生成数据**、**训练** 和 **评估与可视化**。

> 详细的代码结构、每个参数的含义、输出图解读与常见问题见 [GUIDE.md](GUIDE.md)。

## 依赖

```bash
python -m pip install numpy matplotlib torch
```

PyTorch 请按 [官方安装说明](https://pytorch.org/get-started/locally/)选择 CPU / CUDA 版本。SSIM 已内置在 `train.py`，不需要 `torchmetrics` 或 `pytorch-msssim`。

## 1. 按 NF 生成数据

```bash
python generate.py --samples 1000 --out data/v1
```

- 七组菲涅耳数 `0.01, 0.03, 0.1, 0.3, 1, 3, 5` 共用同一批光阑和同一个 train/val/test 划分。
- `NF < --far-threshold`（默认 0.1）用夫琅禾费 FFT 生成标签，其余用缩放角谱。
- 写入数据前自动做传播验证：远场 FFT/角谱一致性、补零与细分收敛、少量抽样点的独立 RS1 积分；报告写入 `validation.json`，不通过则不写训练数组。
- `D` 为光阑非零像素包围盒的较长跨度，`z=D²/(λNF)`，观察面像素间距 `λz/(NΔx)`；默认 256 网格、10 μm 像素、532 nm。

输出：

```text
data/v1/
├── apertures.npy          # (N, grid, grid)，七组共享
├── splits.npz             # train / val / test 编号
├── manifest.json          # 最后写；status=complete 才能训练
├── validation.json
├── propagation_check.png
├── propagation_metrics.csv
└── nf_0p01/ ... nf_5/
    ├── patterns.npy       # 该 NF 的 log 标签
    └── preview.png
```

## 2. 训练

```bash
python train.py --data data/v1 --out runs/v1
```

- 每个 NF 组分别训练 `physics`（固定传播层 + 零初始化残差 U-Net）和 `baseline`（纯 U-Net）。
- 损失：`L1 + 0.1·log 频谱 L1 + λ_s·(1-SSIM) + 1e-3·|ΔI|`；`--lambda-ssim` 默认 0，开启后使用内置 SSIM。
- `--propagator fraunhofer`（默认）/ `asm` / `hybrid`；`hybrid` 在切换点以下用 FFT、以上用 ASM。
- 最优权重只按验证集空间 L1 选择，epoch 0 也参与。

每个模型目录 `runs/v1/nf_*/{physics,baseline}/`：

```text
best_model.pt   history.csv   selection.json   test_metrics.json
loss_curve.png  train_result.png
```

只跑一组或换层对照：

```bash
python train.py --data data/v1 --groups 1 --out runs/nf1
python train.py --data data/v1 --groups 3 5 --models physics \
    --propagator asm --epochs 10 --batch-size 4 --out runs/asm_v1
```

## 3. 评估与可视化

```bash
python evaluate.py --data data/v1 --run runs/v1 --out results/eval_v1
```

默认在同一 test 划分上比较 `fraunhofer_only`、`asm_only`、`physics`、`baseline`。省略 `--run` 则只评估纯物理参照。

输出：

| 文件 | 内容 |
|------|------|
| `summary.csv` | 每个 NF、每种方法的 MAE/RMSE/PSNR/Corr |
| `per_sample.csv` | 逐样本指标 |
| `metrics.txt` | 七组平均表与简短结论 |
| `test_mae_comparison.png` | 测试 MAE 随 NF 变化 |
| `samples.png` | 每个 NF 一个样本：光阑、两种纯物理、两个模型、标签 |
| `error_map.png` | 单样本误差、模型差值、中心剖面、像素误差直方图 |
| `error_histogram.png` | 逐样本 MAE 分布与 physics/baseline 散点 |
| `extreme_test.png` | 单缝、双缝、圆孔、方孔、光栅、环、全开、全挡 |
| `sweep_slit_width.png` 等 | 缝宽、圆孔半径、光栅周期扫描 |

极端形状与扫描默认取 NF=1 组，可用 `--shape-nf` 指定。

## 项目结构

```text
Aperture-Diffraction/
├── README.md
├── GUIDE.md             # 详细代码结构与运行说明
├── .gitignore
├── generate.py          # 按 NF 生成数据 + 传播验证
├── train.py             # 模型、损失、训练与曲线/样例图
├── evaluate.py          # 纯物理/模型对比 + 全部可视化
├── data/.gitkeep        # 生成的数据（不提交）
├── runs/.gitkeep        # 权重与训练记录（不提交）
└── results/.gitkeep     # 评估与图片（不提交）
```

`data/`、`runs/`、`results/` 只保留空目录占位，实际数组、权重、图片由 `.gitignore` 排除。所有输出目录必须不存在；重跑请换名字。设备默认 `auto`，可加 `--device cpu` 或 `--device cuda`。

## 流程检查

用 64×64、24 个光阑走一遍三条命令：

```bash
python generate.py --grid 64 --samples 24 --out data/demo
python train.py --data data/demo --base-ch 8 --batch-size 4 --epochs 1 --out runs/demo
python evaluate.py --data data/demo --out results/demo
```

小网格只用于检查流程，不能作为模型性能结论。

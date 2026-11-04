# Aperture-Diffraction

光阑形状 → 衍射图样的正向预测。正向流程保留三个脚本，分别负责 **按 NF 生成数据**、**训练** 和 **评估与可视化**；逆向只做单张图推理，不训练逆网络；反设计（`design.py`）走相反方向：给定想要的图样，直接解出一块能产生它的光阑。

> 详细的代码结构、每个参数的含义、输出图解读与常见问题见 [GUIDE.md](GUIDE.md)；单图逆向见 [INVERSE.md](INVERSE.md)；目标图样反设计光阑见 [DESIGN.md](DESIGN.md)。

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
- 每张光阑先量支撑宽度 `D`（非零像素包围盒的较长跨度，含孔间距），菲涅耳数 `NF = D²/(λz)`；生成时按目标 NF 反解 `z = D²/(λNF)`，观察面像素间距 `λz/(NΔx)`；默认 256 网格、10 μm 像素、532 nm。

NF 的定义、约定与数值示例见 [GUIDE.md §3.2](GUIDE.md#32-菲涅耳数与传播几何)。

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

- 默认每个 NF 组只训练 `physics`（固定传播层 + 零初始化残差 U-Net）；`baseline`（纯 U-Net）是需要对照时再显式启用的可选项。
- 损失：`L1 + 0.1·log 频谱 L1 + λ_s·(1-SSIM) + 1e-3·|ΔI|`；`--lambda-ssim` 默认 0，开启后使用内置 SSIM。
- `--propagator fraunhofer`（默认）/ `asm` / `hybrid`；`hybrid` 在切换点以下用 FFT、以上用 ASM。
- 最优权重只按验证集空间 L1 选择，epoch 0 也参与。

默认模型目录 `runs/v1/nf_*/physics/`；显式训练 baseline 时才会多出 `baseline/`：

```text
physics/ 或 baseline/
best_model.pt   history.csv   selection.json   test_metrics.json
loss_curve.png  train_result.png
```

只跑一组、启用 baseline 或换层对照：

```bash
python train.py --data data/v1 --groups 1 --out runs/nf1
python train.py --data data/v1 --models physics baseline --out runs/with_baseline
python train.py --data data/v1 --groups 3 5 --models physics \
    --propagator asm --epochs 10 --batch-size 4 --out runs/asm_v1
```

## 3. 评估与可视化

```bash
python evaluate.py --data data/v1 --run runs/v1 --out results/eval_v1
```

默认在同一 test 划分上比较 `fraunhofer_only`、`asm_only`、`physics`；如果 `--run` 中存在 baseline 权重，再一并比较 `baseline`。省略 `--run` 则只评估纯物理参照。

输出：

| 文件 | 内容 |
|------|------|
| `summary.csv` | 每个 NF、每种方法的 MAE/RMSE/PSNR/Corr |
| `per_sample.csv` | 逐样本指标 |
| `metrics.txt` | 七组平均表与简短结论 |
| `test_mae_comparison.png` | 测试 MAE 随 NF 变化 |
| `samples.png` | 每个 NF 一个**不同生成策略**的样本：光阑、标签、各方法预测 |
| `diversity.png` | **八种光阑生成策略**各取样本：光阑、标签、各方法预测 |
| `error_map.png` | 单样本误差、模型差值、中心剖面、像素误差直方图 |
| `error_histogram.png` | 逐样本 MAE 分布；有 baseline 时额外显示对比散点 |
| `extreme_test.png` | 单缝、双缝、圆孔、方孔、光栅、环、全开、全挡 |
| `sweep_slit_width.png` 等 | 缝宽、圆孔半径、光栅周期扫描 |

极端形状与扫描默认取 NF=1 组，可用 `--shape-nf` 指定；`diversity.png` 默认每个策略 1 个样本，
可用 `--diversity-per-strategy N` 增加。光阑类别由 `generate.strategy_assignment(samples, seed)`
从 manifest 复现，不需要改数据格式。

## 4. 单张图逆向（不训练）

```bash
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern path/to/pattern.npy \
  --out inverse_runs/case_nf1
```

- **网络输入只有衍射图样**（`--pattern`）；配对光阑只能通过 `--expected` 用于作图对比，绝不进入 loss/梯度/优化；
- **不需要传 z**：默认 `--geometry auto`，用重建光阑的显著连通域支撑自动估计 z，前 60% 更新、之后冻结；`--geometry canonical` 只是固定参考支撑的受控对照；
- 默认 `--forward-mode checkpoint`，用训练好的 FFT + 残差 U-Net 做正演；`physical` 仅作纯物理对照；
- 损失分阶段：前期只拟合数据/大方向，后期渐入二值/形态学先验；数据项含相关系数项（尺度不变，抑制空/糊斑），先验与数据的占比按 EMA 动态保持；
- 默认迭代 960 步（`--steps` 可继续加长）；收尾默认用 `contour` 对 0.5 等值线做亚像素平滑；`--geometric-stage off` 时才走 `--edge-sigma` 高斯平滑；
- **后段平滑**：默认 `--geom-primitive contour` 对二值像素解提 0.5 等值线，RDP 简化后用亚像素多边形填充，保留任意凹凸/孔洞；`auto` 可改为自动尝试椭圆和正多边形 3/4/5/6，`ellipse` / `polygon --geom-sides N` 可强制指定（`--geometric-stage off` 可关闭）；
- 光阑参数化带平滑水平集；默认不做椭圆/矩形拟合，避免抹掉孔洞；
- 输出 top-3 候选、重建图样、`pattern_mae` / `physical_mae` 等诊断；
- 加 `--expected` / `--expected-sample` 可生成“预期 vs 实际 top1”对照图；参考光阑只做对比，不参与优化；
- 数字 0/8/圆盘/正多边形等测试用例由 `inverse.py --write-cases` 离线生成（GT 仅用于对比，网络不接收光阑）；
- 详细说明见 [INVERSE.md](INVERSE.md)。

## 5. 目标图样 → 反设计光阑（设计，不是反演）

```bash
python design.py --char Z --out design_runs/letterZ_binary
```

- 输入是**想要的**远场图样（字符，或 PNG/JPG/BMP/WEBP/TIFF/NPY 图片，放 `inputs/` 即可），输出是一块**任意形状**的光阑，使它的夫琅禾费衍射图样就是这张图；不要求是“原来那块”光阑；
- 纯 FFT 正向 + 测试时优化，不加载 checkpoint、不用数据集；
- 光阑长什么样不受限，解出来是干涉条纹状的二值掩模（看起来像二维码），`aperture.png` 可直接拿去加工；
- **实振幅光阑的远场必然中心对称** $I(\mathbf{k})=I(-\mathbf{k})$，所以 Z 这种不对称图样不能单独居中：默认把目标放在偏轴 ROI，对跖象限会自动出现旋转 180° 的孪生像（`corr_twin` = 1.0000，物理自检）；
- `--modulation gray` 是连续透过率（拟合上限），`--modulation phase` 是纯相位光阑（Z 居中、无孪生像、效率最高）；
- 输出六联图：目标 / 拟合图样 / 光阑 / 全视场 / 误差 / 收敛曲线；
- 想用自己的图片：把它放进 `inputs/`，然后
  `python design.py --target inputs/你的图.png --out design_runs/我的图`；
  四步操作、图片要求和出错对照表见 [DESIGN.md §3.1](DESIGN.md#31-自己加图片四步)；
- 详细说明见 [DESIGN.md](DESIGN.md)。

## 项目结构

```text
Aperture-Diffraction/
├── README.md
├── GUIDE.md             # 正向流程详细说明
├── INVERSE.md           # 单图逆向说明
├── DESIGN.md            # 目标图样反设计光阑说明
├── .gitignore
├── generate.py          # 按 NF 生成数据 + 传播验证
├── train.py             # 模型、损失、训练与曲线/样例图
├── evaluate.py          # 纯物理/模型对比 + 全部可视化
├── inverse.py           # 单图逆向：正向适配 + 物理先验 + 几何后处理 + 用例生成 + CLI
├── design.py            # 目标图样 → 反设计光阑（二值/灰阶/相位，CGH）
├── data/.gitkeep        # 生成的数据（不提交）
├── runs/.gitkeep        # 权重与训练记录（不提交）
├── results/.gitkeep     # 评估与图片（不提交）
├── inverse_runs/        # 单图逆向输出（不提交）
├── design_runs/         # 反设计输出（不提交）
└── inputs/              # 反设计的目标图样（示例图片）
```

`data/`、`runs/`、`results/`、`inverse_runs/`、`design_runs/` 只保留空目录占位，实际数组、权重、图片由 `.gitignore` 排除。所有输出目录必须不存在；重跑请换名字。设备默认 `auto`，可加 `--device cpu` 或 `--device cuda`。

## 流程检查

用 64×64、24 个光阑走一遍三条命令：

```bash
python generate.py --grid 64 --samples 24 --out data/demo
python train.py --data data/demo --base-ch 8 --batch-size 4 --epochs 1 --out runs/demo
python evaluate.py --data data/demo --out results/demo
```

小网格只用于检查流程，不能作为模型性能结论。

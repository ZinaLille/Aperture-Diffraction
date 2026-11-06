# Aperture-Diffraction

光阑形状 → 衍射图样的正向预测，以及反方向的「给定图样 → 解出光阑」。

- **正向**：`generate.py` 按菲涅耳数造数据 → `train.py` 训练（固定传播层 + 残差 U-Net）→ `evaluate.py` 评估；
- **反设计**：`design.py` 用训练好的 physics+残差 网络当正向，解出一块能产生目标图样的光阑；
- **v2（毛玻璃）**：`generate_v2.py` 在光阑后面加一块随机相位毛玻璃造数据，`train.py` / `evaluate.py` 原样复用。

> 所有命令的速查版（PowerShell 单行、可直接复制）见 [instructions.txt](instructions.txt)。

## 依赖

```bash
python -m pip install numpy matplotlib torch
```

PyTorch 请按 [官方安装说明](https://pytorch.org/get-started/locally/)选择 CPU / CUDA 版本。SSIM 已内置在 `train.py`，不需要 `torchmetrics` 或 `pytorch-msssim`。

## 1. 按 NF 生成数据（v1）

```bash
python generate.py --samples 1000 --out data/v1
```

- 七组菲涅耳数 `0.01, 0.03, 0.1, 0.3, 1, 3, 5` 共用同一批光阑和同一个 train/val/test 划分。
- `NF < --far-threshold`（默认 0.1）用夫琅禾费 FFT 生成标签，其余用缩放角谱。
- 写入数据前自动做传播验证：远场 FFT/角谱一致性、补零与细分收敛、少量抽样点的独立 RS1 积分；报告写入 `validation.json`，不通过则不写训练数组。
- 每张光阑先量支撑宽度 `D`（非零像素包围盒的较长跨度，含孔间距），菲涅耳数 `NF = D²/(λz)`；生成时按目标 NF 反解 `z = D²/(λNF)`，观察面像素间距 `λz/(NΔx)`；默认 256 网格、10 μm 像素、532 nm、`D` 随样本变化所以 `z` 也逐样本不同。
- **`--hologram-fraction 0.2`**：把这一比例的样本替换成**设计产物风格**的密集干涉条纹掩模（周期几个像素、fill 0.3~0.55），
  用来让网络覆盖 `design.py` 查询的那片区域——这是改善「设计结果」最直接的办法（原因见 §4）。
  `--hologram-mode binary|gray|both`、`--hologram-fill LO HI` 控制形态；被替换的编号写进 `manifest.json` 的 `hologram.ids`。
  **未被替换的样本与同 seed/samples 的旧数据集逐位相同**（用独立 RNG，自然光阑生成序列不变）。

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
- 损失：`L1 + λ_f·log 频谱 L1 + λ_s·(1-SSIM) + λ_r·|ΔI|`；`--lambda-freq` 默认 0.1、`--lambda-ssim` 默认 0。
  做 v2（毛玻璃）时改用 `--lambda-freq 0.02`：玻璃残差比 v1 的离散化残差大一个量级以上，
  0.1 时频率项会占掉九成以上损失、空间 L1 几乎学不动。
- `--propagator fraunhofer`（默认）/ `asm` / `hybrid`；`hybrid` 在切换点以下用 FFT、以上用 ASM，
  与 `generate.py` / `generate_v2.py` 生成标签的路径一一对应（做 v2 时建议用它）。
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
python train.py --data data/v1 --groups 3 5 --models physics --propagator asm --epochs 10 --batch-size 4 --out runs/asm_v1
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

## 4. 目标图样 → 反设计光阑（正向 = 训练好的网络）

```bash
python design.py --checkpoint runs/v1/nf_1/physics/best_model.pt --char Z --out design_runs/letterZ_net
```

- 输入是**想要的**衍射图样（字符，或 PNG/JPG/BMP/WEBP/TIFF/NPY 图片，放 `inputs/` 即可），
  输出是一块光阑；优化器的正向就是你训练好的 `physics` 模型（固定传播层 + 残差 U-Net），
  所以毛玻璃、探测器响应之类被网络学进去的系统偏差会一并被考虑；
- 测试时优化：不训练、不改网络，只反传梯度到光阑；解出来是干涉条纹状的二值掩模（可直接拿去加工）；
- **务必加 `--check-glass`**：再放回真实解析模型（玻璃 + 缩放角谱）复算一遍。
  网络只是"训练分布附近的预测器"，而设计优化会主动把光阑推出那个分布，所以它会**系统性高估**：

  | 场景 | 网络预测 corr | 真实模型 corr |
  |------|-------------:|-------------:|
  | 无玻璃（`runs/v1` + 全 0 相位复核） | 0.849 | **0.715** |
  | 固定弱玻璃（v2 模型 + 真实相位复核） | 0.929 | **0.706** |
  | 固定 moderate 玻璃 | 0.95 | 0.17 |

  `--check-glass` 给**全 0 的相位屏**就是"无玻璃"复核；真实实验里没有 φ，这一步用实测图样对比代替。
- `--modulation gray` 是连续透过率（拟合上限），`--modulation phase` 是纯相位光阑；
- 输出 `comparison.png`：给了 `--check-glass` 时是 **8 联图**——
  ① 目标　② **网络推理的图样**（训练时不知道 φ）　③ **物理计算的真实图样**（用真实毛玻璃 φ 算）
  ④ 两者之差　⑤ 光阑　⑥ 全视场（网络）　⑦ 全视场（物理）　⑧ 收敛曲线；
  没给 `--check-glass` 时退回 6 联图（只剩网络侧）；
- 输出：`target.npy` `aperture.npy` `aperture.png` `pattern_pred.npy` `pattern_fit.png` `pattern_roi.png`
  `target_roi.png` `history.npy` `comparison.png` `result.json`
  （给了 `--check-glass` 时还有 `pattern_check_intensity.npy` / `pattern_check_roi.png`）。

## 5. 阶段二 v2：光阑 + 一块固定的随机相位毛玻璃

只新增 `generate_v2.py`；`train.py` / `evaluate.py` 原样复用，`design.py` 也不用改。

```bash
python generate_v2.py --samples 1000 --glass-shared --out data/v2_shared
python train.py       --data data/v2_shared --propagator hybrid --lambda-freq 0.02 --out runs/v2
python evaluate.py    --data data/v2_shared --run runs/v2 --out results/eval_v2
python design.py      --checkpoint runs/v2/nf_1/physics/best_model.pt --char Z --check-glass data/v2_shared/glass_phase.npy --out design_runs/netZ
```

### 5.1 物理模型

```text
光阑 A(x)            毛玻璃 e^{iφ(x)}                     观察面
   │                     │                                  │
   ├──── 同一平面 ───────┤──────── 自由传播 z ──────────────►│
```

```text
I_glass(x) = | P_z[ e^{iφ(x')} · A(x') ](x) |²     → 峰值归一化 + log1p(100·I/peak)，与 v1 完全一致
```

- `P_z` 是 `generate.py` 里那个已验证的缩放角谱；`generate_v2.scaled_field` 是它的**复数版本**
  （`angular_spectrum_raw` 只返回强度，这里要保留相位才能先乘相位屏再传播）；
- φ 是高斯随机相位屏：白噪声 → 高斯低通（滤波核 std = ℓ/2）→ 归一化到指定 rms，自相关 1/e 半径正好是 ℓ；
- 玻璃放在**物面**（紧贴光阑的薄相位屏），这是薄散射体的标准模型；放到中间面会改变观察面网格里 z 的定义，
  当前参数下 2 mm 间距对图样的影响只有 ~1e-5，不值得引入那个歧义；
- 光阑与 v1 同 seed/samples 时**逐位相同**（实测 `array_equal=True`），所以能按样本号对照「加玻璃 / 不加玻璃」；
- **输出目录与 `generate.py` 完全同构**（`apertures.npy` / `splits.npz` / `nf_*/patterns.npy` / `manifest.json`），
  所以 `train.py`、`evaluate.py` 一行都不用改；网络看不到 φ。

### 5.2 用法

```bash
# 一块固定玻璃（整个数据集共用同一块相位屏）——主用例
python generate_v2.py --samples 1000 --glass-shared --out data/v2_shared

# 玻璃强弱：weak / moderate / strong / groundglass
python generate_v2.py --samples 1000 --glass-shared --glass-preset weak --out data/v2_weak

# 按实际玻璃给物理参数（相关长度 µm、相位 rms rad）
python generate_v2.py --samples 1000 --glass-shared --glass-correlation-um 600 --glass-rms-rad 0.5 --out data/v2_custom

# 只要部分 NF 组（更快）
python generate_v2.py --samples 600 --nf 1 --glass-shared --out data/v2_nf1

# 混入设计产物分布（推荐，能改善 design 的可靠性）
python generate_v2.py --samples 1000 --nf 1 --glass-shared --glass-preset weak --hologram-fraction 0.2 --out data/v2_nf1_holo

# 逐样本换玻璃（默认），做「每张图都换一块」的对照
python generate_v2.py --samples 1000 --out data/v2_random
```

### 5.3 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--glass-shared` | 关 | 整个数据集共用一块玻璃（“一块特定的毛玻璃”） |
| `--glass-preset` | `moderate` | `weak` / `moderate` / `strong` / `groundglass` |
| `--glass-correlation-um` | 预设 | 相位相关长度（µm），覆盖预设 |
| `--glass-rms-rad` | 预设 | 相位起伏 rms（弧度），覆盖预设 |
| `--save-clean` | 关 | 额外存无玻璃图样（默认 1000/seed42 下与 `data/v1` 逐位相同，一般不用开） |

预设按「相关长度 / 掩模尺寸」给，64 与 256 网格落在同一个定性区间：

| 预设 | 相关长度（256 网格） | rms | 散射光晕 | 观感 |
|------|---------------------|-----|---------|------|
| `weak` | ≈800 µm | 0.6 rad | ≈3 px | 轻微畸变 |
| `moderate` | ≈300 µm | 1.2 rad | ≈8 px | 明显散斑 |
| `strong` | ≈150 µm | 2.3 rad | ≈17 px | 图样被淹没 |
| `groundglass` | ≈40 µm | 3.0 rad | ≈64 px | 真毛玻璃 |

### 5.4 输出

```text
data/v2_shared/
├── apertures.npy          # (N, grid, grid)；同 seed/samples 时与 v1 逐位相同
├── glass_phase.npy        # (grid, grid) 弧度，那块玻璃；--glass-shared 时是这个形状
├── splits.npz             # train / val / test
├── manifest.json          # config / groups / samples / seed / split_counts —— v1 读的就是这些
├── validation.json  glass_metrics.csv  glass_check.png
└── nf_0p01/ ... nf_5/
    ├── patterns.npy       # 含玻璃的标签（GT）
    └── preview.png
```

### 5.5 生成前验证

| check | 含义 | 阈值 |
|-------|------|------|
| `glass_rms` | 实际相位 rms 与请求是否一致 | 5% |
| `glass_correlation` | 实际自相关 1/e 半径与请求是否一致 | 35%（受像素离散限制） |
| `glass_far_asm_agreement` | 玻璃后 FFT / 缩放角谱给同一张图（切换点） | MAE < 5e-3 |
| `zero_glass_identity` | φ≡0 时含玻璃真值必须**逐位等于**无玻璃图样 | max < 1e-6 |
| `glass_has_effect` | 玻璃确实改变了图样（防止参数写错成空操作） | MAE > 1e-6 |

`info` 里还给出 `grain_px`（散斑颗粒 = 掩模尺寸/D）和 `halo_px`（光晕 = 掩模尺寸/ℓ）。
`grain_px < 1.5` 说明扰动细于探测网格的 Nyquist，会很难学；`weak` 预设下约 `grain≈2 px, halo≈3 px`。

### 5.6 实测

条件：256×256，600 光阑（472/64/64），NF=1，一块固定玻璃，40 epoch（`--propagator hybrid --lambda-freq 0.02`）。

| 玻璃 | 方法 | test MAE | corr |
|------|------|---------:|-----:|
| `weak`（ℓ≈0.79 mm，rms 0.6） | `asm_only`（纯物理） | 9.3365e-4 | 0.9529 |
| `weak` | `physics`（物理层+残差） | **8.4726e-4 / 7.7063e-4** | **0.9635 / 0.9677** |
| `moderate`（ℓ≈0.31 mm，rms 1.2） | `asm_only` | 3.8850e-3 | 0.7434 |
| `moderate` | `physics` | **2.9244e-3** | **0.8893** |

- `weak` 两次训练分别 **−9.3% / −17.5%**，`moderate` 参考值 **−24.7%**；
  cuDNN 非确定性 + 数据集小，单次波动不小，比较方案时请多跑几个 seed；
- 玻璃越强，可学的量越大；不管强弱，残差补的都是“玻璃让图样偏离无玻璃预测”的那部分。

### 5.7 反设计（穿过玻璃）

反设计统一用 `design.py`（正向 = 训练好的模型，不需要 φ）：

```bash
python design.py --checkpoint runs/v2/nf_1/physics/best_model.pt --char Z --check-glass data/v2_shared/glass_phase.npy --out design_runs/netZ
```

实测（同一块弱玻璃、同一个 Z）：网络预测 0.929、真实玻璃 0.706；无玻璃时预测 0.849、真实 0.715。
Z 认得出，但网络会高估 0.13–0.22，所以**必须**用 `--check-glass` 复核。
想更可信，做法是"测量在环"：设计 → 加工/实测 → 把实测图样加进数据集再训一轮。

### 5.8 常见问题

**Q：真的不用改 `train.py`？**
实测跑过：`train.py` 和 `evaluate.py` 直接读 `data/v2_shared`，没有报错。
注意 `--glass-shared` 生成的 `glass_phase.npy` 是 `(grid, grid)`，训练时 v1 根本不读它。

**Q：仿真数据和以后采集的真实数据怎么对齐？**
保持同一套物理参数（grid / 像素 / 波长 / NF 分组）和同一套目录结构；
真实数据的图样替换 `nf_*/patterns.npy`，光阑替换 `apertures.npy`，`splits.npz` 重新划分。
毛玻璃参数按实际玻璃量级给，别用 `strong` 去对一块弱玻璃。

**Q：`--save-clean` 会占多少空间？**
每个 NF 组多一个 `(N, grid, grid)` float32：1000 样本、256 网格约 262 MB，七组约 1.8 GB。
默认 1000 样本 / seed 42 下它与 `data/v1/nf_*/patterns.npy` **逐位相同**，已有 `data/v1` 时完全不用开。

## 项目结构

```text
Aperture-Diffraction/
├── README.md
├── instructions.txt     # 所有命令的 PowerShell 速查（单行、可直接复制）
├── .gitignore
├── generate.py          # 按 NF 生成数据 + 传播验证
├── train.py             # 模型、损失、训练与曲线/样例图
├── evaluate.py          # 纯物理/模型对比 + 全部可视化
├── generate_v2.py       # v2 数据：光阑 + 一块固定的随机相位毛玻璃 → 含扰动的衍射图样
├── design.py            # 目标图样 → 反设计光阑（正向 = 训练好的 physics+残差 网络）
├── data/.gitkeep        # 生成的数据（不提交）
├── runs/.gitkeep        # 权重与训练记录（不提交）
├── results/.gitkeep     # 评估与图片（不提交）
├── design_runs/         # 反设计输出（不提交）
└── inputs/              # 反设计的目标图样（示例图片）
```

`data/`、`runs/`、`results/`、`design_runs/` 只保留空目录占位，实际数组、权重、图片由 `.gitignore` 排除。所有输出目录必须不存在；重跑请换名字。设备默认 `auto`，可加 `--device cpu` 或 `--device cuda`。

## 流程检查

**NF=1 最短跑通**（24 个光阑、5 epoch，约 3 分钟，只为确认四个脚本能串起来）：

```bash
python generate.py --samples 24 --nf 1 --skip-validation --out data/nf1_min
python train.py --data data/nf1_min --groups 1 --base-ch 8 --batch-size 4 --epochs 5 --out runs/nf1_min
python evaluate.py --data data/nf1_min --run runs/nf1_min --out results/eval_nf1_min
python design.py --checkpoint runs/nf1_min/nf_1/physics/best_model.pt --char Z --steps 300 --restarts 1 --out design_runs/nf1_min
```

**64×64 冒烟**（25 秒，只验证代码路径）：

```bash
python generate.py --grid 64 --samples 24 --out data/demo
python train.py --data data/demo --base-ch 8 --batch-size 4 --epochs 1 --out runs/demo
python evaluate.py --data data/demo --out results/demo
```

小网格 / 少样本只用于检查流程，不能作为模型性能结论。
要"能看的指标"，用 [instructions.txt](instructions.txt) 第 4 节的 NF=1 可用版（600 光阑、40 epoch）或毛玻璃版。

# 代码结构与运行说明

这份文档解释 `Aperture-Diffraction` 当前阶段一正向训练工程。读者按顺序读完后，应能独立完成：

1. 按菲涅耳数（NF）生成光阑与衍射图样数据；
2. 逐 NF 训练 `physics`；需要纯网络对照时再显式启用 `baseline`；
3. 生成指标、样本对比图、误差图、极端形状图与参数扫描图。

---

## 0. 五分钟快速开始

```bash
# 依赖
python -m pip install numpy matplotlib torch

# 1. 生成数据（64×64 小规模先走通流程）
python generate.py --grid 64 --samples 24 --out data/demo

# 2. 训练（1 轮即可）
python train.py --data data/demo --base-ch 8 --batch-size 4 --epochs 1 --out runs/demo

# 3. 评估与可视化
python evaluate.py --data data/demo --run runs/demo --out results/demo
```

三条命令跑完后，重点查看：

```text
data/demo/manifest.json
runs/demo/nf_1/physics/loss_curve.png
runs/demo/nf_1/physics/train_result.png
results/demo/samples.png
results/demo/error_map.png
results/demo/extreme_test.png
```

小网格只用于检查流程，不能作为模型性能结论。

---

## 1. 三个脚本的职责

| 脚本 | 输入 | 输出 | 一句话说明 |
|---|---|---|---|
| [`generate.py`](generate.py) | 光阑数、NF 列表、物理参数 | `data/` 数据集 | 光阑 → 传播 → log 标签 |
| [`train.py`](train.py) | `generate.py` 的数据目录 | `runs/` 权重与训练记录 | 每个 NF 训练 physics；baseline 可选 |
| [`evaluate.py`](evaluate.py) | 数据 + 训练输出 | `results/` 指标与图片 | 纯物理与冻结权重对比、可视化 |

数据流：

```text
generate.py
   │  apertures.npy + splits.npz + nf_*/patterns.npy + manifest.json
   ▼
data/v1/
   │
   ▼
train.py
   │  nf_*/physics/best_model.pt + history.csv + 训练图
   │  （显式启用 baseline 时才有 nf_*/baseline/）
   ▼
runs/v1/
   │
   ▼
evaluate.py
   │  summary.csv + per_sample.csv + metrics.txt + 全部评估图
   ▼
results/eval_v1/
```

---

## 2. 目录结构

```text
Aperture-Diffraction/
├── README.md
├── GUIDE.md              # 本文档
├── .gitignore
├── generate.py           # 阶段一：按 NF 生成数据 + 传播验证
├── train.py              # 阶段一：模型、损失、训练与训练图
├── evaluate.py           # 阶段一：评估与全部可视化
├── data/.gitkeep         # 生成的数据（不提交）
├── runs/.gitkeep         # 权重与训练记录（不提交）
└── results/.gitkeep      # 指标与图片（不提交）
```

`data/`、`runs/`、`results/` 只保留目录占位。实际数组、权重和图片由 `.gitignore` 排除，但目录本身保留，脚本可以直接写入。

---

## 3. `generate.py`：按 NF 生成数据

### 3.1 内部结构

| 区块 | 内容 | 关键位置 |
|---|---|---|
| 物理参数与传播 | `PhysicsConfig`、几何关系、FFT、缩放角谱、RS1 抽样 | [generate.py:42](generate.py#L42) |
| 光阑几何 | 八类光阑绘制函数与 `make_aperture` | [generate.py:184](generate.py#L184) |
| 数据划分 | `split_ids` | [generate.py:439](generate.py#L439) |
| 传播验证 | `validate` | [generate.py:473](generate.py#L473) |
| 数据生成 | `generate` 与 CLI | [generate.py:579](generate.py#L579) |

### 3.2 物理约定

对每张光阑：

$$
D=\max(\text{行跨度},\text{列跨度})\cdot\Delta x
$$

$$
z=\frac{D^2}{\lambda N_F},\qquad
\Delta x_{\text{out}}=\frac{\lambda z}{N\Delta x_{\text{in}}}
$$

- `D` 是光阑非零像素包围盒的较长跨度，包含孔间距，不是单个孔径。
- 同一 NF 组内不同光阑的 `D` 不同，所以每个样本的 `z` 不同。
- 观察面是随样本缩放的网格，不是固定像素的相机。
- 默认：`256×256`、输入像素 `10 μm`、波长 `532 nm`。

标签处理：

$$
I_{\text{norm}}=\frac{I}{\max I},\qquad
I_{\text{label}}=\frac{\log(1+\alpha I_{\text{norm}})}{\max\log(1+\alpha I_{\text{norm}})}
$$

默认 `α=100`，输出在 `[0,1]`。

### 3.3 两种标签传播

| NF | 标签方法 |
|---|---|
| `NF < 0.1` | 夫琅禾费 FFT |
| `NF ≥ 0.1` | 缩放角谱 ASM |

- 0.1 是本实验的数值切换点，不是普适物理分界。
- 角谱默认 2 倍子像素、2 倍补零。
- 两种方法使用同一套观察面坐标，便于比较。

### 3.4 八类光阑

```text
multi_elements  多个矩形、圆、椭圆或多边形
slits           单缝、多缝
grating         有限尺寸光栅
hole_array      网格、环形或随机孔洞
letter          T、L、H、X、O、C、F、E
annular         环、多环、扇形
mixed_grating   光栅 + 缺陷 + 附加圆孔
polygons        正多边形、不规则多边形、星形、十字
```

每类先在 256×256 的 legacy 几何上绘制，再缩放到 `--grid`。

### 3.5 数据划分

`split_ids()` 按光阑类别分层：

- 每类约 10% 进 test；
- 再 10% 进 val；
- 其余进 train。

1000 个样本约为 `808 / 96 / 96`；24 个样本为 `8 / 8 / 8`。七组共用同一划分。

### 3.6 生成前传播验证

`validate()` 检查：

1. 远场 FFT 与缩放角谱在 `NF < 0.1` 时的一致性；
2. 角谱补零加倍、子像素加密后的收敛；
3. 少量观察点上的独立 Rayleigh–Sommerfeld 积分；
4. 切换点两侧 FFT/ASM 的差异。

输出：

- `validation.json`
- `propagation_check.png`
- `propagation_metrics.csv`

验证不通过时直接退出，不写训练数组。调试时可加 `--skip-validation`，但正式数据不建议跳过。

### 3.7 输出结构

```text
data/v1/
├── apertures.npy          # (N, grid, grid) float32，七组共享
├── splits.npz             # train / val / test 编号
├── manifest.json          # 最后写；status=complete 才能训练
├── validation.json
├── propagation_check.png
├── propagation_metrics.csv
└── nf_0p01/ ... nf_5/
    ├── patterns.npy       # 该 NF 的 log 标签
    └── preview.png        # 八类光阑示例
```

- 数组用 memmap 写入；1000 个 256² 样本约 2.1 GB。
- `manifest.json` 最后写；生成中断时训练会拒绝启动。

### 3.8 常用命令

```bash
# 正式数据
python generate.py --samples 1000 --out data/v1

# 快速检查
python generate.py --grid 64 --samples 24 --out data/demo

# 只生成一个 NF
python generate.py --nf 3 --samples 1000 --out data/nf3

# 自定义 NF 组
python generate.py --nf 0.01 0.03 0.1 0.3 1 3 5 --samples 1000 --out data/v1
```

### 3.9 参数

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--grid` | 256 | 图像边长，必须能被 8 整除 |
| `--samples` | 1000 | 原始光阑数，七组共用 |
| `--nf` | 七组 | Fresnel 数，范围 `(0,5]`，不能重复 |
| `--pitch-um` | 10 | 输入像素间距 μm |
| `--wavelength-nm` | 532 | 波长 nm |
| `--far-threshold` | 0.1 | FFT / ASM 标签切换点 |
| `--oversample` | 2 | 角谱子像素倍数 |
| `--pad-factor` | 2 | 角谱补零倍数 |
| `--seed` | 42 | 光阑与划分种子 |
| `--out` | `data/v1` | 输出目录，必须不存在 |

---

## 4. `train.py`：逐 NF 训练

### 4.1 内部结构

| 区块 | 内容 | 关键位置 |
|---|---|---|
| 损失与 SSIM | 内置 SSIM、`Loss` | [train.py:57](train.py#L57) |
| 网络 | `ConvBlock`、`ConditionedUNet` | [train.py:118](train.py#L118) |
| 物理层 | `FraunhoferDetectorLayer`、`AngularSpectrumDetectorLayer` | [train.py:151](train.py#L151) |
| 组合模型 | `ForwardModel` | [train.py:227](train.py#L227) |
| 数据 | `GroupData`、条件标准化 | [train.py:278](train.py#L278) |
| 训练 | `run_group` | [train.py:484](train.py#L484) |
| CLI | `main` | [train.py:558](train.py#L558) |

### 4.2 physics（默认）与 baseline（可选）

**physics（默认训练）**

```text
光阑 a ──► 固定传播层 Φ(a,z) ──► I_sim ──┐
   │                                      │
   └──► 残差 U-Net ──► ΔI ────────────────┴──► Î = I_sim + ΔI
```

- 输入 4 通道：光阑、`I_sim`、两个条件通道；
- 残差输出层权重和偏置初始化为 0；
- epoch 0 时 `ΔI=0`，预测严格等于纯物理结果。

**baseline（可选对照）**

```text
光阑 a ──► 纯 U-Net ──► Î
```

- 仅在显式传 `--models baseline` 或 `--models physics baseline` 时训练；
- 输入 3 通道：光阑、两个条件通道；
- 没有物理层；
- 输出层偏置初始化为训练标签均值。

条件通道：

```text
c = [log(z), log(detector_pitch)]
```

使用训练集统计量标准化后，展开成两张常量图。

### 4.3 U-Net 结构

- 3 次下采样；
- 通道数 `base_ch → 2× → 4× → 8×`；
- 每级 `Conv → GroupNorm → GELU → Conv → GroupNorm → GELU`；
- 转置卷积上采样 + 跳跃连接；
- 默认 `base_ch=32`，输出 1 通道。

### 4.4 损失

$$
\mathcal L=
\underbrace{\|\hat I-I\|_1}_{\text{空间 L1}}
+\lambda_f\underbrace{\|\log(1+|\mathcal F\hat I|)-\log(1+|\mathcal F I|)\|_1}_{\text{频域幅度}}
+\lambda_s\underbrace{(1-\mathrm{SSIM})}_{\text{结构}}
+\lambda_r\underbrace{\|\Delta I\|_1}_{\text{残差正则}}
$$

默认：

- `λ_f = 0.1`
- `λ_s = 0.0`，可开启；SSIM 是内置实现，不依赖 `torchmetrics` / `pytorch_msssim`
- `λ_r = 0.001`，只对 physics 生效

频域项是输出图样再做 FFT，不是物理传播场的频谱。

### 4.5 训练与选优

- 优化器：AdamW，`lr=3e-4`，`weight_decay=1e-4`；
- 余弦退火；
- 梯度裁剪 1.0；
- 默认 50 epochs、batch 16；
- 先评估 epoch 0 并保存 checkpoint；
- 之后只有验证集空间 L1 更低才覆盖 `best_model.pt`；
- 训练完成后加载最佳权重，再在 test 上评估。

如果 `selection.json` 的最佳轮次是 0，说明网络没有超过纯物理起点。

### 4.6 输出

```text
runs/v1/
├── run_config.json
├── summary.csv
└── nf_0p01/ ... nf_5/
    └── physics/                 # 默认
        ├── best_model.pt
        ├── history.csv
        ├── selection.json
        ├── test_metrics.json
        ├── loss_curve.png
        └── train_result.png

# 显式启用 baseline 时，每个 NF 组会多出：
# baseline/
#   └── ... 同上
```

`best_model.pt` 里保存了模型结构信息、`physics_config`、条件标准化统计、`propagator` 与 NF，评估时据此重建模型并校验。

### 4.7 常用命令

```bash
# 七组 physics（默认）
python train.py --data data/v1 --out runs/v1

# 只跑 NF=1
python train.py --data data/v1 --groups 1 --out runs/nf1

# 需要纯网络对照时，同时训练 physics + baseline
python train.py --data data/v1 --models physics baseline --out runs/with_baseline

# 只训练 baseline
python train.py --data data/v1 --models baseline --out runs/baseline

# 角谱物理层换层对照
python train.py --data data/v1 --groups 3 5 --models physics \
    --propagator asm --epochs 10 --batch-size 4 --out runs/asm_v1

# 按切换点自动选 FFT / ASM
python train.py --data data/v1 --propagator hybrid --out runs/hybrid
```

### 4.8 参数

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--data` | `data/v1` | `generate.py` 的输出目录 |
| `--out` | `runs/<时间戳>` | 新目录，必须不存在 |
| `--groups` | 全部 | 只训练指定 NF |
| `--models` | physics | 默认只训练 physics；需要对照时加 baseline |
| `--propagator` | fraunhofer | physics 分支的传播层 |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 16 | 显存不足时改 4/2 |
| `--base-ch` | 32 | U-Net 基础通道 |
| `--lr` | 3e-4 | 学习率 |
| `--lambda-ssim` | 0 | 开启后使用内置 SSIM |
| `--workers` | 0 | Windows 建议保持 0 |
| `--device` | auto | `auto/cpu/cuda` |

---

## 5. `evaluate.py`：评估与可视化

### 5.1 内部结构

| 区块 | 内容 | 关键位置 |
|---|---|---|
| 权重加载 | `load_model` | [evaluate.py:37](evaluate.py#L37) |
| 方法集合 | 纯 FFT、纯 ASM、physics；baseline 可选 | [evaluate.py:61](evaluate.py#L61) |
| 逐样本评估 | `evaluate_group` | [evaluate.py:74](evaluate.py#L74) |
| 基础图 | MAE 曲线、样本图、误差图、直方图 | [evaluate.py:99](evaluate.py#L99) |
| 极端形状 | 八种标准形状 | [evaluate.py:256](evaluate.py#L256) |
| 参数扫描 | 缝宽、圆孔半径、光栅周期 | [evaluate.py:328](evaluate.py#L328) |
| CLI | `main` | [evaluate.py:377](evaluate.py#L377) |

### 5.2 比较的方法

固定比较：

- `fraunhofer_only`：纯夫琅禾费 FFT；
- `asm_only`：纯缩放角谱。

提供 `--run` 时再加载：

- `physics`：训练好的物理 + 残差模型；
- `baseline`：如果该 run 里存在 `baseline/best_model.pt`，一并加载。

`load_model` 会检查：

- checkpoint 与数据集的 `PhysicsConfig` 是否一致；
- checkpoint 的 NF 与当前组是否一致；
- 条件标准化 mean/std 是否与当前训练划分一致。

不一致会直接报错，避免拿错权重评估。

### 5.3 指标

逐样本、逐像素计算：

$$
\mathrm{MAE}=\frac1P\sum|\hat I-I|,\qquad
\mathrm{RMSE}=\sqrt{\frac1P\sum(\hat I-I)^2}
$$

$$
\mathrm{PSNR}=20\log_{10}\frac{1}{\mathrm{RMSE}},\qquad
\mathrm{Corr}=\frac{\mathrm{Cov}(\hat I,I)}{\sigma_{\hat I}\sigma_I}
$$

PSNR 使用固定数据范围 1，所以接近零误差时会出现很大的 dB 值。

- `summary.csv`：每个 NF、每种方法一行；
- `per_sample.csv`：每个测试样本、每种方法一行；
- `metrics.txt`：七组平均与简短结论。

### 5.4 图怎么看

**`samples.png`**

每个 NF 一行，列为：

```text
光阑 | 纯 FFT | 纯 ASM | physics | [baseline] | 标签
```

`baseline` 列只在训练过 baseline 时出现。

重点看：

- 低 NF 组：FFT 应该接近标签；
- 高 NF 组：ASM 应该接近标签，FFT 误差更大；
- physics 应该接近它的物理层；
- 如果启用了 baseline，它通常更模糊或偏差更大。

**`error_map.png`**

针对 `--shape-nf` 最近的一组，展示：

```text
光阑 | 物理层/参照 | 预测 | 真值
|误差| | 方法差值 | 中心行剖面 | 像素误差直方图
```

如果同时有 physics 和 baseline，中间是 `baseline error - physics error`，绿色表示 physics 更好。

**`error_histogram.png`**

- 逐样本 MAE 分布；
- 有 baseline 时，额外给出 physics 相对 baseline 的改善百分比；
- 有 baseline 时，额外给出 physics vs baseline 散点。

**`test_mae_comparison.png`**

测试 MAE 随 NF 的变化，对数纵轴。适合说明远场近似在哪个 NF 开始失效。

**`extreme_test.png`**

八种形状：

```text
单缝、双缝、圆孔、方孔、光栅、环、全开、全挡
```

每个形状按自身 `D` 计算 `z`，保证都在指定 NF 下；列是光阑 + 各方法预测。

**三张 sweep 图**

```text
sweep_slit_width.png
sweep_circle_radius.png
sweep_grating_period.png
```

上半行是图样，下半行是中心行剖面；优先使用训练好的 physics，没有则退回纯 ASM。

### 5.5 常用命令

```bash
# 完整对比
python evaluate.py --data data/v1 --run runs/v1 --out results/eval_v1

# 只评估纯物理
python evaluate.py --data data/v1 --out results/physics

# 只评估部分 NF
python evaluate.py --data data/v1 --run runs/v1 --groups 3 5 --out results/nf35

# 使用验证集
python evaluate.py --data data/v1 --run runs/v1 --split val --out results/val

# 指定极端形状/扫描的 NF 组
python evaluate.py --data data/v1 --run runs/v1 --shape-nf 5 --out results/nf5
```

### 5.6 参数

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--data` | `data/v1` | 数据集 |
| `--run` | 无 | `train.py` 输出目录；省略则只做纯物理 |
| `--out` | `results/eval_v1` | 输出目录，必须不存在 |
| `--groups` | 全部 | 只评估指定 NF |
| `--split` | test | `val` 或 `test` |
| `--batch-size` | 8 | 显存不足时减小 |
| `--shape-nf` | 1.0 | 极端形状/扫描选择最近的 NF 组 |
| `--device` | auto | `auto/cpu/cuda` |

---

## 6. 推荐工作流

### 6.1 环境检查

```bash
python generate.py --grid 64 --samples 24 --out data/demo
python train.py --data data/demo --base-ch 8 --batch-size 4 --epochs 1 --out runs/demo
python evaluate.py --data data/demo --run runs/demo --out results/demo
```

### 6.2 正式阶段一

```bash
python generate.py --samples 1000 --out data/v1
python train.py --data data/v1 --out runs/v1
python evaluate.py --data data/v1 --run runs/v1 --out results/eval_v1
```

### 6.3 单 NF 快速实验

```bash
python generate.py --nf 3 --samples 1000 --out data/nf3
python train.py --data data/nf3 --out runs/nf3
python evaluate.py --data data/nf3 --run runs/nf3 --out results/nf3
```

### 6.4 角谱物理层对照

```bash
python train.py --data data/v1 --groups 3 5 --models physics \
    --propagator asm --epochs 10 --batch-size 4 --out runs/asm_v1

python evaluate.py --data data/v1 --run runs/asm_v1 --groups 3 5 \
    --out results/asm_v1
```

---

## 7. 如何判断结果是否合理

按这个顺序看：

1. `results/.../summary.csv`：纯 FFT、纯 ASM、physics（若启用则加 baseline）的 MAE 是否呈预期关系；
2. `samples.png`：每行 NF 对应的图样形状是否正确；
3. `selection.json`：最佳轮次是否为 0；若为 0，说明残差网络没有超过物理起点；
4. `loss_curve.png`：训练/验证 L1 是否稳定，频域项与残差项是否发散；
5. `train_result.png`：残差是否集中在物理层误差较大的区域；
6. `extreme_test.png` 与 sweep：形状/参数变化时图样是否连续、无明显伪影。

预期现象：

- `NF < 0.1`：标签由 FFT 生成，`fraunhofer_only` 接近机器零；
- `NF ≥ 0.1`：标签由 ASM 生成，`asm_only` 接近机器零；
- 默认 physics 使用 FFT，在高 NF 组会与 ASM 标签有明显偏差；
- 换成 `--propagator asm` 后，physics 的 epoch 0 通常就能接近标签；
- 如果显式训练了 baseline，它没有物理约束，通常误差更大。

---

## 8. 常见问题

**输出目录已存在**

三个脚本都要求输出目录不存在。换 `--out` 名字，避免覆盖已有结果。

**训练提示数据集不完整**

`manifest.json` 不存在或 `status != complete`。重新运行 `generate.py`，不要手工改 manifest。

**checkpoint 与数据配置不一致**

通常是数据生成和训练使用了不同的 `--grid`、`--pitch-um` 或 `--wavelength-nm`。训练和评估必须使用同一份数据。

**CUDA 显存不足**

先降 batch：

```bash
python train.py ... --batch-size 4
```

或降模型：

```bash
python train.py ... --base-ch 16 --batch-size 2
```

也可以 `--device cpu`。

**小网格上光阑消失**

`--grid 32` 时细小结构可能消失，脚本会报错。流程检查建议至少 64；正式数据用 256。

**为什么 physics 在高 NF 变差**

默认 `--propagator fraunhofer`，高 NF 标签是 ASM，FFT 物理层存在近似误差。改用：

```bash
--propagator asm
```

或：

```bash
--propagator hybrid
```

`hybrid` 在 `NF < 0.1` 用 FFT，其余用 ASM。

---

## 9. 最小命令总结

```bash
python generate.py --samples 1000 --out data/v1
python train.py --data data/v1 --out runs/v1
python evaluate.py --data data/v1 --run runs/v1 --out results/eval_v1
```

三条命令覆盖阶段一正向训练的全部工作。

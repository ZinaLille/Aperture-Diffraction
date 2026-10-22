# 单张衍射图样反演光阑

这套代码只做一件事：

```text
输入一张衍射图样 + 冻结的 physics 正向 checkpoint
        ↓ 测试时优化，不训练任何逆网络
输出 top-3 光阑候选 + 重建图样 + 诊断指标
```

原有 `generate.py` / `train.py` / `evaluate.py` 和 `runs/v1/**/best_model.pt`
保持不变；逆向逻辑全部收在单个 `inverse.py` 里，按顺序分为四块：

| `inverse.py` 内部区块 | 作用 |
|---|---|
| 1. 正向算子 | checkpoint、`FrozenForward`、自洽 z、图样读写 |
| 2. 几何后处理 | 连通域清理、椭圆/矩形拟合 |
| 3. 反演核心 | 配置、物理先验、`PixelField`、多尺度/多起点、top-3 |
| 4. 输出与 CLI | 画图、保存、单图推理入口 |

不训练逆网络，不使用成对训练数据，不做数据集批量评估。
`inverse.py` 只 import `generate.py` 和 `train.py` 里的正向定义，不复制正向网络。

---

## 1. z 未知时怎么处理

当前 `runs/v1` 的 physics checkpoint 全部使用 `fraunhofer` 传播层。
`train.py` 的 `ForwardModel` 只有在：

```python
propagator == "asm" and kind == "physics"
```

时才用 ASM，否则用 FFT。因此：

- `z` 不进入 FFT 主传播项；
- `z` 只进入残差网络的条件
  `condition = [log(z), log(detector_pitch)]`；
- NF<0.1 时残差为零，`z` 完全不重要；
- NF≥0.1 时用当前光阑的支撑 `D` 和已知 NF 自洽计算：

```text
D = max(行跨度, 列跨度) × input_pitch
z = D² / (λ · NF)
detector_pitch = λz / (N · input_pitch)
condition = (log([z, detector_pitch]) - mean) / std
```

CLI 的 `--geometry` 有三种：

| 模式 | 含义 |
|---|---|
| `self-consistent`（默认） | 每 10 步用当前二值光阑的支撑更新 `z`，不需要外部给 z |
| `canonical` | 用 `D_ref = support_fraction × N × input_pitch` 固定一个参考 z |
| `known` | 用户用 `--z` 明确给出传播距离（米） |

只有 `--forward-mode physical`（纯 ASM）时 `z` 才真正进入传播；这时未知 z 会带来
`a(x) → a(x/s), z → s²z` 的尺度耦合，必须用 `--z` 或接受“只恢复形状、不恢复绝对尺寸”。

---

## 2. 反演损失

```text
L = ||F(a) - y||₁
  + 0.1 · ||log(1+|FFT(F(a))|) - log(1+|FFT(y)|)||₁
  + λ_tv    · TV(a)
  + λ_lap   · Laplacian(a)
  + λ_perim · TV(a) / area
  + λ_bin   · mean(4a(1-a))
  + λ_open  · Opening₃(a)
  + λ_open5 · Opening₅(a)
  + λ_island· Island(a)
  + λ_area  · AreaHinge(a)
  + λ_res   · mean|delta|
```

| 项 | 物理含义 |
|---|---|
| 数据 L1 | 重建图样必须匹配观测图样 |
| log 频谱 L1 | 约束条纹/高频结构，和正向训练标签域一致 |
| `TV` | 各向同性总变差，抑制棋盘格与孤立噪点，近似边界周长 |
| `Laplacian` | 二阶差分，惩罚尖刺、细丝和锯齿边界 |
| `TV/area` | 周长/面积比，扭曲/破碎形状比值高；权重小，避免误伤光栅/字母 |
| `4a(1-a)` | 二值先验，把透过率推向 0/1 |
| `Opening₃` | 软形态学开运算，惩罚宽/高小于约 3 像素的细碎结构 |
| `Opening₅` | 更严格的 5 像素最小特征；默认关闭，避免擦掉细缝 |
| `Island` | 按面积归一化的孤立小岛惩罚：局部 7×7 前景质量不足的像素 |
| `AreaHinge` | 只惩罚极端全开/全挡（面积 <0.002 或 >0.6） |
| `mean|delta|` | checkpoint 模式下抑制反演利用残差网络 |

参数化本身也加了物理约束：

```text
a = sigmoid(GaussianBlur(theta, sigma) / T)
```

`GaussianBlur` 是固定核，不是训练层；它限制边界曲率和最小特征尺寸，减少细丝/尖刺。
默认 `sigma=1.0`，可用 `--smooth-sigma` 调整。

最后还有一步**几何参数整理**（`inverse.py` 的“2. 几何后处理”区块）：

```text
二值 mask
  → 4-邻域连通域
  → 删除面积 < min_area 的小岛
  → 对面积 <= max_fit_area、solidity >= min_solidity 的组件拟合椭圆/旋转矩形
  → 大结构/带孔结构保留原始像素
```

默认 `--primitive-fit ellipse --primitive-min-area 20 --primitive-max-fit-area 4000
--primitive-min-solidity 0.2`。这样输出的小孔、小元素天然是有限尺寸、平滑边界的椭圆，
不会留下扭曲的细丝或随机小岛。

不加强制单连通、强制居中、强制对称等先验，因为数据里本来就有多元素、偏心和带孔的光阑。

`--lambda-area-center` 默认 0.1，用对数面积把 soft 面积轻轻拉向该 restart 的初始面积，
减少“面积塌缩/膨胀”的局部极小；它不是硬约束。

---

## 3. 单图推理流程

1. 读取 pattern，归一化到 `[0,1]`（PNG 可 `--input-mode image` 做 peak + `log1p`）；
2. 加载 physics checkpoint，冻结全部参数；
3. 初始化 4 个候选：稀疏孔、随机块、椭圆 blob、光栅/环等不同面积与几何类型；
4. 多尺度 `64 → 128 → 256` 优化；
5. 最后 80 步切 STE，直接优化二值 mask；
6. 对二值 mask 做几何整理：删除小岛，把中小型连通域拟合成椭圆/旋转矩形（`--primitive-fit`）；
7. 每个 restart 用整理后的光阑重算 `z`，计算：
   - `pattern_mae`：checkpoint 重建误差；
   - `physical_mae`：纯 FFT/ASM 独立重建误差；
   - `residual_ratio`：残差项占比；
   - `condition_zscore`：条件是否外推；
   - `area_fraction` / `support_fraction` / `primitive_components`；
8. 按 `data_loss + 0.05·prior` 排序，贪心选 3 个质心对齐 IoU < 0.8 的不同候选；
9. 保存 top-3 和 `comparison.png`。

---

## 4. 用法

```bash
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern path/to/pattern.npy \
  --nf 1 \
  --out inverse_runs/case_nf1
```

也可以从 `runs/v1` 自动找 checkpoint：

```bash
python inverse.py \
  --run runs/v1 --nf 1 \
  --pattern path/to/pattern.npy \
  --out inverse_runs/case_nf1
```

已知传播距离时：

```bash
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern path/to/pattern.npy \
  --nf 1 --geometry known --z 0.12 \
  --out inverse_runs/case_nf1
```

纯物理 ASM 对照：

```bash
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern path/to/pattern.npy \
  --nf 1 --forward-mode physical --geometry known --z 0.12 \
  --out inverse_runs/case_nf1_asm
```

PNG 原始强度图：

```bash
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern path/to/camera.png \
  --nf 1 --input-mode image --invert-image \
  --out inverse_runs/case_png
```

---

## 5. 输出

```text
inverse_runs/case_nf1/
├── result.json
├── config.json
├── comparison.png
├── rank1_aperture.npy
├── rank1_aperture_bin.npy
├── rank1_aperture.png
├── rank1_pattern_pred.png
├── rank1_pattern_physical.png
├── rank1_residual.png
├── rank1_history.csv
├── rank1_loss_curve.png
├── rank2_...
└── rank3_...
```

`result.json` 关键字段：

- `candidates`：每个候选的 `pattern_mae`、`physical_mae`、`residual_ratio`、
  `area_fraction`、`support_fraction`、`z_m`、`condition_zscore`、
  `primitive_components`、`primitive_info`；
- `pairwise_iou` / `ambiguous`：top-3 是否给出不同解；
- `warnings`：condition 外推、残差过大、纯物理不一致、面积异常。

`primitive_info` 记录连通域总数、拟合数、删除的小岛数、保留的大结构数。
如果 `primitive_info.fallback = kept_raw_pixel_mask`，说明几何拟合后为空，保留了像素解。

无 GT 时判断结果是否可信，优先看：

1. `pattern_mae` 是否接近正向模型自身的 test MAE（NF=1 约 5e-4 量级）；
2. `physical_mae` 与 `pattern_mae` 是否同量级；
3. `residual_ratio` 是否过大；
4. top-3 的 `pairwise_iou` 是否很低（多解明显）；
5. `area_fraction` 是否落在合理范围（数据里约 0.006–0.44）。

---

## 6. 已知边界

- 单张强度图丢失相位，反演是严重欠定的：远场存在平移、镜像/共轭多解；
  近场未知 z 时存在尺度耦合；
- 简单圆孔可以从随机初始化精确恢复；复杂多孔/环形/光栅会陷入
  “能拟合图样但形状不同”的局部极小；
- 现在输出会先做几何整理：小岛删除、中小连通域拟合椭圆/矩形。复杂多孔目标
  不再输出扭曲细丝/散点，而是一组有限尺寸的椭圆孔；但孔位/孔数仍可能与真值不同；
- 因此输出 top-3 和 `pattern_mae` / `physical_mae`，而不是承诺唯一真值；
- 首版没有训练逆网络，也没有用数据集做批量评估；所有计算都在单张图上完成。

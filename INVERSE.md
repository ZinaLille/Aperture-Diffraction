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

`geometry_from_mask`（[inverse.py:209](inverse.py#L209)）就是这个式子；NF 的定义、符号约定与数值示例见 [GUIDE.md §3.2](GUIDE.md#32-菲涅耳数与传播几何)。

---

## 2. 反演损失：先拟合大方向，后收紧细节

### 2.1 总损失

```text
L = L_data + w_coarse(t) · L_coarse
  + g(t) · Σ_d w_d(t) · λ_d · P_d(a)
  + λ_res · mean|delta|

L_data   = ||F(a) - y||₁ + 0.02·||log(1+|FFT(F(a))|) - log(1+|FFT(y)|)||₁
           + 0.5·(1 - corr(F(a), y))
L_coarse = ||AvgPool₉(F(a)) - AvgPool₉(y)||₁
P_d      = TV、Laplacian、TV/area、mean(4a(1-a))、Opening₃、Opening₅、
           Island、AreaHinge、AreaCenter
```

- `w_d(t)`：每个先验项自己的时间窗（0→1 平滑过渡），决定“什么时候开始管形状”；
- `w_coarse(t)`：低频数据项权重，只在前段有值，先定大方向；
- `g(t)`：先验总权重的自适应增益，让“先验 / 空间 L1”的占比贴住 `--prior-ratio`（默认 0.25）：

```text
g(t)       = clamp( ratio(t) · EMA(L1) / EMA(Σ λ_d P_d) , 0, 50 )
ratio(t)   = prior_ratio · smoothstep((t - fit) / (prior - fit))
```

`λ_d` 仍是各项之间的相对比例，`g(t)` 只统一缩放总强度；这样数据项变小时先验不会被动变大。

相关系数项 `1 - corr` 是尺度不变的：空光阑或大光斑即使绝对 L1 很小，相关系数也低，
会被这项压住；log 频谱项权重从 0.1 降到 0.02，避免它压过空间拟合。两者都可用
`--lambda-corr` / `--lambda-freq` 调。

### 2.2 三阶段调度

| 阶段 | 进度 | 数据项 | 形状先验 | 平滑尺度 | 目的 |
|---|---|---|---|---|---|
| 拟合 | `[0, fit)` = `[0, 0.30)` | L1 + 相关 + 频谱 + 低频粗项 | 全 0 | 1.6σ₀ | 只拟合大方向，形状不受约束 |
| 渐入 | `[fit, prior)` = `[0.30, 0.60)` | 同上，粗项退场 | TV/Lap/面积/面积中心先到满；perim/island 稍后 | 1.2σ₀ | 结构先验渐入 |
| 细节 | `[prior, detail)` = `[0.60, 0.85)` | 同上 | 二值/Opening₃/Opening₅ 渐入 | σ₀ | 收紧具体形状 |
| 收尾 | `[detail, 1]` = `[0.85, 1]` | 同上 | 全部满权重 | σ₀ | 平滑收尾（不锐化） |

`fit`、`prior`、`detail` 分别由 `--fit-fraction`、`--prior-fraction`、`--detail-fraction`
控制；平滑尺度以 `--smooth-sigma`（记为 σ₀）为基准，粗阶段乘 `--sigma-coarse-scale`，
细节阶段乘 `--sigma-detail-scale`。σ₀ 在 64/128/256 网格上分别对应 4/2/1 个输出像素的平滑，
所以三个阶段本身就是从“粗形状”到“细节边缘”的过渡。

默认迭代轮次是 `(240, 240, 480)`（共 960 步，比旧版 440 步更长），可用 `--steps` 继续加长；
`--temperature-end`（默认 0.35）控制 sigmoid 最终软硬，`--edge-sigma`（默认 1.5）控制收尾平滑。

唯一全程生效的形状项是 `AreaHinge`：它只在面积 <0.2% 或 >60% 时发力，用来防止
“前期没有形状先验”时把光阑整块压没或撑满；正常面积下它的值和梯度都是 0，不约束具体形状。

调度实现在 `inverse.py` 的 `loss_factors()`，`run_restart()` 每个 step 计算 gain 并写入
`rank*_history.csv`（`phase`、`gain`、`target_ratio`、`prior_ratio`、`f_tv`、`f_bin`、`sigma`）。
`rank*_loss_curve.png` 下半张画“实际先验/L1 占比 vs 目标占比 + 自适应增益”，可以直接看到权重怎么变。

`--loss-schedule fixed` 回到旧行为：先验全程固定权重、`sigma` 保持 `--smooth-sigma`、
STE 只看最后 80 步，不做 gain 归一化；适合和旧结果做 A/B 对照。

### 2.3 各项物理含义

| 项 | 物理含义 |
|---|---|
| 数据 L1 | 重建图样必须匹配观测图样 |
| log 频谱 L1 | 约束条纹/高频结构，和正向训练标签域一致 |
| 低频粗项 | 前期只看低通图样，避免过早被细节/噪点带偏 |
| `TV` | 各向同性总变差，抑制棋盘格与孤立噪点，近似边界周长 |
| `Laplacian` | 二阶差分，惩罚尖刺、细丝和锯齿边界 |
| `TV/area` | 周长/面积比，扭曲/破碎形状比值高；权重小，避免误伤光栅/字母 |
| `4a(1-a)` | 二值先验，把透过率推向 0/1 |
| `Opening₃` | 软形态学开运算，惩罚宽/高小于约 3 像素的细碎结构 |
| `Opening₅` | 更严格的 5 像素最小特征；默认关闭，避免擦掉细缝 |
| `Island` | 按面积归一化的孤立小岛惩罚：局部 7×7 前景质量不足的像素 |
| `AreaHinge` | 只惩罚极端全开/全挡（面积 <0.002 或 >0.6） |
| `mean|delta|` | checkpoint 模式下抑制反演利用残差网络 |

### 2.4 参数化与几何整理

参数化本身也加了物理约束：

```text
a = sigmoid(GaussianBlur(theta, sigma(t)) / T)
```

`GaussianBlur` 是固定核，不是训练层；它限制边界曲率和最小特征尺寸，减少细丝/尖刺。
`sigma(t)` 按 2.2 的三阶段变化（1.6σ₀ → 1.2σ₀ → σ₀ → σ₀），`T` 从 1.0 退火到 0.35，
所以 sigmoid 本身不会把边缘压成 0/1。

优化结束后还有一步**边缘参数化收尾**（非几何模式）：先删掉小连通域，再对二值 mask 做
`edge_sigma`（默认 1.5 像素）的高斯平滑，最后取 0.5 等值线。保存的
`rank*_aperture.png`、`expected_vs_top1.png` 里的 top1 光阑和最终前向都用这张平滑后的软 mask，
所以边缘是平滑过渡而不是硬阈值锯齿；`edge_sigma=0` 可以关掉。

最后还有一步**几何参数整理**（`inverse.py` 的“2. 几何后处理”区块）：

```text
二值 mask
  → 4-邻域连通域
  → 删除面积 < min_area 的小岛
  → 对面积 <= max_fit_area、solidity >= min_solidity 的组件拟合椭圆/旋转矩形
  → 大结构/带孔结构保留原始像素
```

默认 `--primitive-fit none`：椭圆/矩形拟合会把孔洞和细结构抹掉，只在明确需要“干净几何”时
再打开 `ellipse`/`rect`。打开时 `--primitive-min-area 20 --primitive-max-fit-area 4000
<<<<<<< Updated upstream
--primitive-min-solidity 0.2`。这样输出的小孔、小元素天然是有限尺寸、平滑边界的椭圆，
不会留下扭曲的细丝或随机小岛。
=======
--primitive-min-solidity 0.2`。整理后会重新传播和评分；仅当图样 MAE、数据损失和排名评分
均未变差（容差 1e-8）时才接受，否则保留原始像素候选。不能为使形状好看而强行改成椭圆。
>>>>>>> Stashed changes

不加强制单连通、强制居中、强制对称等先验，因为数据里本来就有多元素、偏心和带孔的光阑。

`--lambda-area-center` 默认 0.1，用对数面积把 soft 面积轻轻拉向该 restart 的初始面积，
减少“面积塌缩/膨胀”的局部极小；它不是硬约束。

### 2.5 后段几何参数化（默认开启）

像素解只作为初始化；优化结束后进入几何阶段：

```text
像素 mask
  → 连通域（正基元）+ 内部孔洞（负基元）
  → 基元参数：ellipse (cx,cy,rx,ry,angle) 或 polygon (cx,cy,R,angle)
  → 用同一前向模型和数据项优化基元参数（geom_steps=300, lr=0.02）
  → 正基元并集 − 孔洞并集，sigmoid(softness) 渲染
```

- `--geom-primitive ellipse`（默认）或 `polygon`；`polygon` 时所有基元共用
  `--geom-sides N`（正 N 边形），参数为 `(cx, cy, R, angle)`；
- 正/负基元最多 `--geom-max-pos 8` / `--geom-max-neg 4`；半径用 softplus 保证为正，
  并限制在画幅内；
- 正则：`geom_support_weight`（支撑外质量）、`geom_area_weight`（对数面积目标），
  防止基元膨胀到全画幅；
- 输出边界由参数化 sigmoid 生成，天然平滑；`--geometric-stage off` 回到像素解；
- `result.json` 的 `primitive_info` 记录 `primitive`、`sides` 和
  `positive_primitives` / `negative_primitives`（像素坐标）。

实测（NF=1，scale 0.5，physical 前向，known z）：正多边形基元对方形/五边形/六边形
IoU≈0.91/0.99/0.998（三角形 0.70，需 `--restarts 8`）；椭圆基元对圆盘 IoU≈1.0。
限制：基元并集能精确表达圆盘、正多边形、多孔组合；但数字 0/8 这类“细环 + 孔”需要
像素初始化里已经有闭合孔，初始化没有孔时几何阶段无法凭空生成，仍会退化成若干基元块。

---

## 3. 单图推理流程

1. 读取图样：`--pattern` 是网络唯一输入（`.npy` 或 PNG，PNG 可 `--input-mode image` 做 peak + `log1p`）；
2. 加载 physics checkpoint，冻结全部参数；
3. 初始化 4 个候选：稀疏孔、随机块、椭圆 blob、光栅/环等不同面积与几何类型；
4. 多尺度 `64 → 128 → 256` 优化，同时按 §2 的三阶段调度：前期只拟合数据/大方向，中期渐入形状先验，后期收紧细节；
5. 最后 160 步且进入细节阶段后切 STE；后续再做 `edge_sigma` 高斯平滑，输出平滑边缘；
6. 对二值 mask 做几何整理：删除小岛，把中小型连通域拟合成椭圆/旋转矩形（`--primitive-fit`）；
7. 每个 restart 用整理后的光阑重算 `z`，计算：
   - `pattern_mae`：checkpoint 重建误差；
   - `physical_mae`：纯 FFT/ASM 独立重建误差；
   - `residual_ratio`：残差项占比；
   - `condition_zscore`：条件是否外推；
   - `area_fraction` / `support_fraction` / `primitive_components`；
8. 按 `data_loss + 0.05·prior` 排序，贪心选 3 个质心对齐 IoU < 0.8 的不同候选；
9. 保存 top-3 和 `comparison.png`；给了 `--expected` / `--expected-sample` 时，
   再用参考光阑生成 `expected_vs_top1.png`。

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

测试用例（数字 0 / 数字 8 / 圆盘）：

```bash
# 离线生成图样 + GT（只写文件，不做反演；GT 只用于作图）
python inverse.py --write-cases inverse_runs/cases --nf 1 --case-scale 0.5

# 反演：网络输入只有 pattern.npy
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern inverse_runs/cases/digit8_nf1_s0.5/pattern.npy \
  --expected inverse_runs/cases/digit8_nf1_s0.5/gt.npy \
  --nf 1 --forward-mode physical --geometry known --z 1.2639 \
  --out inverse_runs/final_digit8
```

- `pattern.npy` 是唯一进入反演的输入；`gt.npy` 只通过 `--expected` 用于
  `expected_vs_top1.png` 和 `reference` 指标，不进入 loss、梯度或优化；
- `--case-scale` 控制光阑大小：远场图样的特征尺寸约为 `N/n_D` 像素，光阑越大图样越紧凑、越不可辨识；
- `--geometry known --z` 是“已知传播距离”的受控对比；真实场景用默认的 self-consistent。

正多边形用例：

```bash
python inverse.py --write-cases inverse_runs/cases --nf 1 --case-scale 0.5 \
  --case-shapes triangle square pentagon hexagon

python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern inverse_runs/cases/hexagon_nf1_s0.5/pattern.npy \
  --expected inverse_runs/cases/hexagon_nf1_s0.5/gt.npy \
  --nf 1 --forward-mode physical --geometry known --z 0.7699 \
  --geom-primitive polygon --geom-sides 6 \
  --out inverse_runs/poly_hexagon
```

`--geom-sides` 要和目标多边形一致；`--geom-primitive polygon` 只影响后段基元拟合，
网络输入仍只有 `pattern.npy`。

带“预期光阑”的对比（只做图，不参与优化）：

```bash
# 直接用 data/v1 里的第 8 个样本作预期光阑
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern inverse_runs/sample8_pattern.npy \
  --nf 1 --dataset data/v1 --expected-sample 8 \
  --out inverse_runs/sample8_cmp
```

```bash
# 或给任意 .npy / .png 的预期光阑（白=开孔）
python inverse.py \
  --checkpoint runs/v1/nf_1/physics/best_model.pt \
  --pattern path/to/pattern.npy \
  --nf 1 --expected path/to/expected_aperture.npy \
  --out inverse_runs/case_cmp
```

`--expected` / `--expected-sample` 只影响最后的 `expected_vs_top1.png` 和 `result.json`
里的 `reference` 数字；它不进入损失、梯度、`z` 更新或任何优化步骤。想复现旧权重行为时加
`--loss-schedule fixed`。

---

## 5. 输出

```text
inverse_runs/case_nf1/
├── result.json
├── config.json
├── comparison.png
├── expected_vs_top1.png          # 仅当给了 --expected* 时
├── expected_aperture_bin.npy
├── expected_pattern_pred.npy
├── reference.json
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

`expected_vs_top1.png` 是 2×4 的对照图：

```text
预期光阑 | top1 光阑 | 叠加（红=预期，绿=top1，黄=重合） | |预期-top1|
预期图样 | 观测图样 | top1 重建图样 | 残差
```

标题里给出 `IoU`、预期图样的 MAE、top1 的 MAE 与两者面积。

`result.json` 关键字段：

- `candidates`：每个候选的 `pattern_mae`、`physical_mae`、`residual_ratio`、
  `area_fraction`、`support_fraction`、`z_m`、`condition_zscore`、
  `primitive_components`、`primitive_info`、`loss_gain`、`coarse_loss`；
- `reference`：预期光阑的 `pattern_mae`、`iou_top1`、`area_fraction`、`z_m`、`source`；
  只做诊断，不参与优化；
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
- 前期没有形状先验时，个别 restart（尤其是稀疏初始化）可能退化成空 mask；
  这种候选面积接近 0，`select_diverse` 会把它过滤掉，必要时加 `--restarts` 留余量；
- 数字 0/8 这类“环 + 孔”形状在 NF=1 的远场 log 图样里与实心圆盘高度接近：
  同一强度的多个孔径都能拟合到很小的 MAE，优化容易落到“实心斑/单孔”的局部极小。
  实测（`--geometry known`，scale 0.5，960 步 + 几何参数化）：圆盘 IoU≈1.0（单椭圆，pattern MAE 2.2e-4），数字 8 约 0.18、数字 0 约 0.04；
  这是单幅远场强度 + 无逆网络先验的固有限制，不是输入接口问题；
- 要稳定恢复 0/8 这类拓扑，需要训练 pattern→aperture 的逆网络（数据里有 1000 对），
  或引入多平面/相位/支撑约束；只调 loss 权重收益有限；
- 简单实心圆孔可以从随机初始化恢复；复杂多孔/环形/光栅会陷入
  “能拟合图样但形状不同”的局部极小；
- 默认不再做椭圆/矩形拟合（会把孔洞抹掉）；需要干净几何时显式加 `--primitive-fit ellipse`；
- 因此输出 top-3 和 `pattern_mae` / `physical_mae`，而不是承诺唯一真值；
- 首版没有训练逆网络，也没有用数据集做批量评估；所有计算都在单张图上完成。

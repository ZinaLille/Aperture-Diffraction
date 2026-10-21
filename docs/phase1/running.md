# 阶段一：运行与审计说明

阶段一只有三个入口，均在项目根目录运行：

| 命令 | 作用 |
|------|------|
| `python -m aperture.generate` | 传播验证 + 生成七组菲涅耳数数据 |
| `python -m aperture.train` | 训练 physics / baseline 模型 |
| `python -m aperture.evaluate` | 纯物理参照、模型对比；加 `--rs1` 做独立审计 |

三种等价写法：`python -m aperture.generate`（项目根目录）、`python -m aperture generate`（统一入口）、`python aperture/generate.py`（直接运行，会自动切回项目根目录）。

## 1. 生成数据

```bash
python -m aperture.generate --samples 1000 --out data/forward/v1
```

- 生成前自动做传播验证（圆孔、单缝、双缝、光栅；补零/加密/RS1 抽查），报告写入 `validation.json`，不通过就不写数据。
- `--samples` 是原始光阑数，七组共用同一批光阑；1000 个 256² 样本的全部数组约 4.2 GB。
- 输出：`manifest.json`（最后写）、`apertures.npy`、`fraunhofer_patterns.npy`、`splits.npz`、`nf_*/{patterns.npy,patterns_linear.npy,metadata.json,preview.png}`。
- 物理约定：`D` 是光阑包围盒长轴跨度，`z=D²/(λNF)`，观察面像素间距 `λz/(NΔx)`；默认 256 网格、10 μm 像素、532 nm。
- `NF<0.1` 用夫琅禾费 FFT 生成标签，其余用缩放角谱；0.1 是本实验的方法切换点。

## 2. 训练

先做只读检查（不创建输出目录）：

```bash
python -m aperture.train --data data/forward/v1 --dry-run
```

先跑一组，再跑全部：

```bash
python -m aperture.train --data data/forward/v1 --groups 1 --out runs/forward/nf1_v1
python -m aperture.train --data data/forward/v1 --out runs/forward/v1
```

- 每个 NF 组独立训练 `physics`（物理层 + 残差）和 `baseline`（纯网络），默认 50 轮、batch 16、学习率 3e-4。
- 最优权重只按验证集空间 L1 选择，epoch0 也参与；测试集选完权重后才评估。
- 输出在 `runs/forward/v1/nf_*/{physics,baseline}/`：`best_model.pt`、`history.csv`、`selection.json`、`test_metrics.json`、`timing.json`、`loss_curve.png`、`val_samples.png`。
- 角谱换层对照：

```bash
python -m aperture.train --data data/forward/v1 --groups 3 5 --models physics \
    --propagator asm --epochs 10 --batch-size 4 --out runs/forward/asm_v1
```

## 3. 评估

```bash
python -m aperture.evaluate --data data/forward/v1 \
    --run runs/forward/v1 --out results/forward/eval_v1
```

- 在同一划分和观察坐标下比较纯夫琅禾费、纯角谱、physics、baseline。
- 输出 `summary.csv`、逐样本 CSV、`samples.png`、`test_mae_comparison.png`、`implementation_checks.json`、`report.md`。
- 省略 `--run` 就只评估纯物理参照。

独立 RS1 审计：

```bash
python -m aperture.evaluate --rs1 --data data/forward/v1 \
    --run runs/forward/v1 --asm-run runs/forward/asm_v1 \
    --out results/forward/rs1_v1
```

- 八类各 3 个新光阑，NF=0.03/3/5，共 72 例；RS1 直接积分按 2→4→8→16 倍逐级加密。
- 主指标用同一个 RS 抽样峰值作分母；网络的归一化 log 指标只作辅助，不能当绝对强度结论。
- 输出 `cases/*.npz`、`case_metrics.csv`、`convergence.csv`、`comparisons.csv`、`summary.csv`、`win_counts.json`、`report.md`。

## 4. 约定

- 输出目录必须不存在（`new_output`）；重跑请换名字，避免覆盖结果。
- 数据 `manifest.json` 最后写；训练和评估都会校验文件哈希、划分隔离与数据身份。
- 测试：`python -m unittest tests.test_physics -v`（11 项）。
- 设备：`--device auto/cpu/cuda`；显存不足时用 `--batch-size 4`。

# 第一阶段：多菲涅耳数组正向预测

截至2026-10-16，多菲涅耳数理想正向预测及收益归因阶段已完成。最初的前四步已扩展为：数据与传播验证、原物理层/基线训练、角谱换层、独立RS1审计及训练计时。
默认网络物理层仍为夫琅禾费模型，可显式选择角谱或hybrid；没有进入逆向重建。

公开进度见[阶段检查点](../../docs/phase1/CHECKPOINT.md)，完整指标、图和配置见[轻量结果](../../docs/phase1/README.md)。这些是已完成运行的记录，GitHub不附带工作数据和权重；首次克隆请按[项目主页的完整复现顺序](../../README.md)生成数据并训练。

维护者本机已生成正式首批 `data/forward/v1/`：1000个原始光阑，每组1000对样本，七组共7000对。
每组共享808/96/96的训练/验证/测试划分。已有数据时直接跳到“训练前只读检查”和“自行开始训练”。
重新生成时请把输出改为新目录（如 `data/forward/v2`），不能重复使用v1。

## 原有结果保护

- 早期远场脚本的产物（`data/forward/legacy/`、`runs/forward/legacy/`、`results/forward/legacy/`）不作为本包输出。
- 本包输出统一在 `data/forward/`、`runs/forward/`、`results/forward/checks/` 与 `results/forward/reference/`，均由 `.gitignore` 排除。
- 所有输出根目录必须不存在；没有覆盖开关。再次运行请选择新目录。
- `legacy/generate_data.py` 仅把创建目录和设置随机种子移进 `main()`，避免导入几何函数就创建旧数据目录。旧命令生成逻辑及默认参数不变。

## 物理约定

默认输入为 256×256，输入像素间距 10 μm，波长 532 nm。
`D` 定义为光阑非零像素包围盒中较长的轴向跨度（包含像素宽度及孔间距），不是单孔半径。
`NF=D²/(λz)`，按每个样本的 D 和指定 NF 计算 z。八种几何策略保留，数量平衡；第一轮不加入乘性缩放或加性噪声。
新实验局部修正多缝中心的偏移方向，避免旧辅助函数沿长轴排列导致多缝合并；旧函数和旧结果保持原样。

| NF | 标签方法 |
|---|---|
| 0.01、0.03 | 夫琅禾费 FFT |
| 0.10、0.30、1、3、5 | 缩放角谱参考计算 |

NF 必须在 (0,5]；0.1 是本实验的方法切换点，不是普适物理分界。

### 观察面不是固定相机

选择 `detector_pitch = λz/(N*input_pitch)`，使输出坐标为 `x_j=(j-N/2)*detector_pitch`。
同一个样本的 FFT 与角谱输出严格使用相同观察位置。
但不同样本/距离的观察面像素间距不同：这是缩放观察面实验，不能解释成固定像素、固定视场的相机随 z 移动。
每个样本记录 D、z、NF、观察面像素间距、视场和标签方法。神经网络同时输入训练集归一化后的 log(z) 和 log(detector_pitch)。

### 角谱参考的实现与边界

传播频域传递函数为 `H_AS=exp(ikz*sqrt(1-λ²(fx²+fy²)))`。
把它分解为菲涅耳传递函数及精确非近轴色散修正，先在频域施加修正，再通过缩放菲涅耳积分求输出。
这保留了角谱色散关系，避免远距离原生角谱 FFT 需要巨大的周期窗口。
默认输入像素作分片常量解释，2 倍子像素积分、2 倍补零，忽略倏逝谱。
它仍有有限窗口和离散积分误差，不宣称连续精确解；验证包括补零加倍、积分网格加密、独立 RS1 积分抽查。
参考与远场点采样 FFT 还存在像素面积积分差异，验证报告会显示这些误差。

## 运行环境

需要 Python、numpy、matplotlib、torch。
本机已验证Python 3.11.16、NumPy 1.23.5、Matplotlib 3.7.2、PyTorch 2.14.0+cu130，可运行RTX 5070 Laptop GPU的FFT、卷积和反向传播。
其他设备请按[官方安装说明](https://pytorch.org/get-started/locally/)选择PyTorch构建；不要把仅识别到CUDA视为内核兼容性验证。
本机已有环境可激活：

```powershell
conda activate d2l
```

其他机器可创建自己的环境，并安装项目根目录`requirements.txt`中的依赖。
训练入口实际探测 CUDA 内核；auto 模式失败时回退 CPU，显式 `--device cuda` 失败则报错。CPU 可用 `--device cpu`。
默认两种模型均使用 `L1 + 0.1*频域log幅度L1`，SSIM 默认共同关闭。
若给 `--lambda-ssim 0.2`，须安装 torchmetrics 或 pytorch-msssim；请求 SSIM 但依赖不可用时会报错，而不是静默改变配置。

以下命令在项目根目录运行；无需执行旧 generate_data.py/train.py/baseline.py。

## 1. 传播验证

```powershell
python -m src.forward.validate --out results/forward/checks/v1
```

输出 `validation.json`、逐形状指标 CSV、传播图及误差随 NF 的曲线。
只有验证通过且配置完全匹配，生成器才会接受报告。
默认验收阈值：远场/切换点 log图 MAE<0.005，补零/加密 log图 MAE<0.002，独立积分抽查的峰值归一化最大误差<2%。
这些是数值验收阈值，不是预测性能承诺或普适物理阈值。

## 2. 生成数据

```powershell
python -m src.forward.generate --samples 1000 --validation results/forward/checks/v1/validation.json --out data/forward/v1
```

1000 是原始光阑数量，同一批光阑共享到七个 NF 组，各组分别有 1000 对输入/标签。
需要更大训练集可改为 `--samples 5000`；先小规模确认磁盘和运行时间。1000 个 256² 样本的全部新数组约占 4.2 GB 十进制空间。
不传 `--validation` 时，生成器会在新数据目录内先执行同样的验证；失败则不生成训练数组。

```text
data/forward/v1/
  manifest.json                 # 完成标志、物理配置、分组、输入/划分哈希
  validation.json
  apertures.npy                 # 各组共享输入 [N,256,256]
  fraunhofer_patterns.npy       # 纯物理参照，同一缩放坐标下归一化图样
  splits.npz                    # 原始光阑ID的train/val/test，所有组共享
  nf_0p01/ ... nf_5/
    patterns.npy                # 与共享输入同一行ID对应的log标签
    patterns_linear.npy         # 相对单位的线性强度；未假定物理光功率标定
    metadata.json               # 每个输入ID的传播/观察面参数
    preview.png
```

分层按八类形状划分约80/10/10（每类至少1个验证和测试样本）。同一原始光阑在任何 NF 组的归属都不变。
完整 `manifest.json` 只在全部生成成功后写入；未完成的数据不能训练。

## 3. 训练前只读检查

```powershell
python -m src.forward.train --data data/forward/v1 --dry-run
```

检查文件身份、划分隔离、输入/标签/元数据ID、模型前向、零残差初始化；打印各组初始纯物理验证误差。
不更新参数、不创建训练输出。
预检查分为两种：Torch/NumPy物理实现一致性要求<1e-5；旧物理与新标签的差异仅报告，绝不作为退出条件。

## 4. 自行开始训练

先运行一组：

```powershell
python -m src.forward.train --data data/forward/v1 --groups 1 --out runs/forward/nf1_v1
```

再运行全部七组：

```powershell
python -m src.forward.train --data data/forward/v1 --out runs/forward/all_v1
```

默认每组分别训练 physics（夫琅禾费+残差）和 baseline（无物理层），各50轮、batch16、base32、学习率3e-4。
两者共享卷积骨干、传播条件输入、数据划分、学习率、训练轮数和公共数据损失；physics另有1e-3残差正则。
可用 `--models physics` 或 `--models baseline` 单独运行；显存不足用 `--batch-size 4`。
各组重新初始化，不能把它解释为“仅远场训练、外推到近场”的实验。

输出纯物理测试指标、每种模型的 epoch0 验证指标、每轮历史、最佳权重、测试逐样本指标、曲线和验证样例图。
最佳模型只根据验证集空间 L1 选择，包括零更新的 epoch0，避免远场本来正确却被强制选成较差模型。
测试集仅在选好权重后评估。模型不裁剪输出，避免隐藏不合理值。
PSNR使用固定数据范围1（不同于旧脚本的预测依赖峰值），不能把新旧PSNR当作完全同定义。
默认命令不改变旧模型设定；可通过文末的 `--propagator` 参数开展角谱物理层对照。

## 小规模复现与测试

```powershell
python -m unittest tests.test_pipeline tests.test_rs1_audit -v
python -m src.forward.validate --grid 64 --out results/forward/checks/my_smoke
python -m src.forward.generate --grid 64 --samples 24 --validation results/forward/checks/my_smoke/validation.json --out data/forward/my_smoke
python -m src.forward.train --data data/forward/my_smoke --base-ch 8 --batch-size 8 --dry-run
```

小网格仅用于检查流程，几何细节可能被下采样改变，不能用于正式性能结论。

## 更新：纯角谱参照与可选角谱训练

新增 `python -m src.forward.reference`，只读现有数据和最佳权重，不训练网络、不写回旧结果。
会在同一划分、观察坐标和后处理下比较纯FFT、纯角谱、已有残差模型和已有基线。

```powershell
conda activate d2l
python -m src.forward.reference --data data/forward/v1 --compare-run runs/forward/all_first --device cuda --out results/forward/reference/asm_v2
```

本次已评估的参照保存在 `results/forward/reference/asm_v1`，上面的v2仅为再次评估示例，不能使用已存在的输出目录。
默认同时评估验证/测试集，角谱用float64、batch4；`--precision float32`可测训练所用精度，`--groups 3 5`可只检查重点组。
输出包括 `summary.csv`、逐样本指标、样例图、实现一致性检查、对比图和总耗时。
前向计时排除预热、数据加载、主机到设备传输、指标计算与绘图，在CUDA边界同步。旧模型用同一批量重新前向测量。
角谱的float64与神经网络的float32数值精度不同，不能把速度表直接当作同精度算法排名。

近场标签本就由角谱算子生成，所以极小误差主要是数值一致性验证，不是学习收益。
远场标签使用点采样FFT，纯角谱包含子像素积分，存在离散差异；建议远场保留FFT。

训练新增参数：

| 参数 | 含义 |
|---|---|
| `--propagator fraunhofer` | 默认，保留旧夫琅禾费层 |
| `--propagator asm` | 所选组全部使用角谱层 |
| `--propagator hybrid` | NF低于数据切换点时用FFT，其余用角谱 |
| `--asm-precision float32` | 默认训练传播精度；相位系数仍用双精度计算 |
| `--asm-precision float64` | 更高精度传播，输出转回网络输入精度 |

角谱层零可学习参数、支持对输入和物理距离自动微分；接收实际单位为米的距离，不把归一化网络条件当距离。
旧权重仍能按默认夫琅禾费模型读取；新权重记录实际propagator和ASM精度。
骨干、损失、输出是否裁剪等保持一致，避免换层时同时改变其他实验因素。

先做只读检查：

```powershell
python -m src.forward.train --data data/forward/v1 --groups 3 5 --models physics --propagator hybrid --batch-size 4 --device cuda --dry-run
```

维护者已完成的重点组配置为ASM、10轮、batch4，两组均选择epoch0。若自行重复这一对照，可使用新的输出目录：

```powershell
python -m src.forward.train --data data/forward/v1 --groups 3 5 --models physics --propagator asm --epochs 10 --batch-size 4 --device cuda --out runs/forward/asm_nf3_nf5_reproduce
```

## 更新：训练时间统计（默认启用）

只影响未来写入新目录的训练，不回写、估算或补填旧训练的耗时。
CPU使用单调墙钟，GPU在计时块前后同步，避免只统计到异步提交时间。

`history.csv`新增列：

| 列 | 口径 |
|---|---|
| train_seconds | 本轮训练，含加载、传输、前向、反向与更新 |
| validation_seconds | 本轮验证，含加载、传输与损失计算 |
| checkpoint_seconds | 本轮最优权重保存时间，未保存则0 |
| epoch_seconds | 训练、验证、选优、调度的本轮墙钟，不含最终写这条日志及控制台打印 |
| elapsed_fit_seconds | 从开始训练循环到记录本轮的累计时间 |

控制台每轮打印训练、验证、单轮和累计时间。
每个模型的 `timing.json` 每轮更新，完成时包含初始化、训练循环、测试、单模型总时长及平均每轮时长。
`fit_wall_seconds`包括循环内日志与权重I/O；`model_total_seconds`还包括初始化、选优、测试和绘图。
每组另有 `group_timing.json`；运行根目录有 `timing_summary.csv` 和 `timing.json`。
根目录总时长从main入口开始，包含数据检查和设备探测，不包含Python解释器启动和模块导入。
各时间口径有包含关系，不能把总时间与它包含的子项再次相加。

## 已完成：扩大独立RS1验证

新增`python -m src.forward.rs1_audit`。在八类各3个新光阑上检查NF=0.03、3、5，共72例；每例73至82个共同观察点，独立RS1积分全部通过加密收敛检查。

主对照为加入解析sinc像素面积因子的夫琅禾费与当前角谱，原点采样FFT另行保留。纯物理指标共用RS抽样峰值；网络归一化log输出的辅助指标分别解释。

使用已生成数据和已训练权重时：

```powershell
python -m src.forward.rs1_audit --data data/forward/v1 --old-run runs/forward/all_first --asm-run runs/forward/asm_nf3_nf5_e10_v1 --out results/forward/checks/rs1_audit_v2 --device cuda
```

如果数据或权重使用其他输出名称，须相应替换三个输入路径。此命令只做冻结评估，不更新权重。详细口径见[审计说明](RS1_AUDIT_README.md)，结果见[公开汇总](../../docs/phase1/README.md)。

NF=3、5的48例均支持角谱更接近独立积分，但这不构成新增网络收益：两组角谱残差最优均为epoch0。下一步固定系统偏差实验尚未开展。

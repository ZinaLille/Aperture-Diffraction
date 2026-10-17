# 本次GitHub更新清单

比较基准：最初阅读README时的远场FFT版本；当前状态：2026-10-16第一阶段完成检查点。

本地目录没有`.git`，因此这是按项目演进和现有文件核对的发布清单，尚未对远端commit做差异比较，也没有执行上传、提交或推送。

## 1. 必须新增的14个Python文件

建议保留完整`fresnel_experiment`包，不要只挑几个入口：

| 文件 | 新增作用 |
|---|---|
| `__init__.py` | 实验包入口 |
| `apertures.py` | 八类光阑复用、修正多缝排列、原始ID分层划分 |
| `physics.py` | 物理尺度、NF、夫琅禾费/缩放角谱/独立RS1及指标 |
| `validate.py` | 标签生成前的传播、补零、积分加密和独立积分检查 |
| `generate.py` | 七组标签、共享输入和划分、线性强度与log标签、元数据 |
| `models.py` | 条件U-Net、基线、可选物理层和零残差初始化 |
| `torch_physics.py` | 可微角谱层，接收以米为单位的实际距离 |
| `train.py` | 分组训练、epoch0选优、测试隔离、计时与新目录保护 |
| `reference.py` | 纯FFT/角谱与已有冻结模型的统一参照评估 |
| `timing.py` | CPU墙钟、CUDA边界同步和时间格式 |
| `rs1_audit.py` | 新光阑、独立RS1、匹配像素面积对照、逐级收敛 |
| `test_pipeline.py` | 数值、梯度、划分、导入副作用和防覆盖检查 |
| `test_rs1_audit.py` | 共同尺度、观察点与独立积分/方孔sinc解检查 |
| `io_utils.py` | 独占输出目录、JSON和SHA256 |

其中`fresnel_experiment/train.py`与根目录`train.py`是不同入口；新实验使用`python -m fresnel_experiment.train`。

## 2. 已有文件的处理

| 文件 | 本次发布动作 |
|---|---|
| 根目录`generate_data.py` | **更新**。前期将随机种子和创建旧data目录移入main，供新包安全导入几何函数；旧命令默认生成逻辑保留 |
| 根目录`train.py` | 保留。新包仍导入其中的`ConvBlock`和`Loss`，不能因为有了新train入口就删掉 |
| `baseline.py`、`evaluate.py` | 保留早期流程；本阶段没有将它们改成角谱实验入口 |
| 根目录`README.md` | **更新**为当前进度、结果、边界和完整复现入口 |
| `fresnel_experiment/README.md` | **新增/更新**，提供工具参数及输出说明 |
| `fresnel_experiment/RS1_AUDIT_README.md` | **新增**，说明独立积分与指标口径 |
| `.gitignore`、`requirements.txt` | **新增**，声明运行输出排除项与核心依赖 |

如果旧GitHub仓库已经包含四个根脚本，通常只需更新`generate_data.py`，并确认另三个仍在；新建仓库时请把四个一起带上。此处“更新”指相对早期版本的变更，本轮没有重新改动物理或训练代码。

## 3. 建议随代码提交的展示材料

- [公开阶段检查点](PHASE1_CHECKPOINT.md)：完成项、结论和后续边界。
- [阶段结果](phase1_results/README.md)及其CSV、JSON、PNG：目前约1.1 MB，不含大数组和权重。
- [早期远场说明](legacy_far_field.md)：保留项目起点，并修正过去的过强解读。
- 本清单及[机器可读文件清单](UPLOAD_MANIFEST.json)。

公开摘要包含原七组结果、角谱替换的epoch0选优、学习曲线、训练计时、独立RS1收敛和来源哈希。GitHub上的README图片与数据链接均指向这份公开材料，不依赖被排除的本机输出目录。

## 4. 本次不直接提交的内容

| 内容 | 原因或替代方式 |
|---|---|
| `data/`、`fresnel_data/`，所有`.npy/.npz` | 可由脚本生成；正式数据约4.2 GB，公开清单和配置已另存 |
| `fresnel_runs/`、所有`.pt/.pth/.ckpt` | 完整权重和每次运行输出留在本机；需要分发时另行规划Release或LFS |
| `fresnel_checks/`、`fresnel_reference/`、`fresnel_analysis/` | 大量重复工作产物；选择的公开指标和图已复制至docs |
| `change_protection/`、`checkpoints/` | 本机备份、绝对路径与完整哈希快照；公开检查点另存docs |
| `fresnel_experiment/legacy_artifacts_before.json`、`verification.json` | 本机审计记录 |
| `fresnel_experiment/IMPLEMENTATION_REPORT.md`、`ASM_TIMING_REPORT.md` | 历史实施快照，含本机路径及当时状态；当前结果已由公开文档整理 |
| 缓存、编辑器设置、`publish_checks/` | 本地开发或发布检查产物 |

GitHub普通Git提交会拒绝超过100 MiB的单文件，浏览器上传的单文件上限为25 MiB；大数据不要直接拖入普通仓库。[GitHub官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

`.gitignore`只影响尚未跟踪的文件，不会自动从已有提交中删除旧权重或旧图。已有远端历史产物本次不做删除或历史重写。[Git官方说明](https://git-scm.com/docs/gitignore)

## 5. README本次怎样体现进度

1. 首屏明确第一阶段完成，给出“原远场版本→当前多NF验证”的变化表。
2. 将“网络预测极准”改为可归因的证据：原物理层误差、旧残差补偿、角谱替换、epoch0以及独立RS1。
3. 分开呈现标签全图log MAE与独立积分抽样强度指标，避免错误比较。
4. 保留逆向重建和鲁棒性目标，同时标明尚未开展。
5. 修正“有物理层就保证最终输出物理正确”“SSIM必须因结构关闭”“纯CNN普遍学不会”等过强结论。
6. 提供在新输出目录中的完整复现命令，说明运行数据/权重不随普通代码仓库发布。

旧README全文备份仅在本机`change_protection/README_before_github_update_20261016.md`，当前阶段检查点不回写。

## 6. 上传前的操作

当前目录不是Git克隆目录。可将清单中的文件复制到你已有的克隆目录，或按原目录结构在GitHub上传；不要把所有工作目录打包上传。

若使用已有Git克隆目录，在那里检查变更，再按文件范围暂存：

```bash
git status --short
git add README.md .gitignore requirements.txt generate_data.py train.py baseline.py evaluate.py fresnel_experiment docs
git diff --cached --stat
git diff --cached --name-only
```

检查暂存内容与`UPLOAD_MANIFEST.json`的清单相符，尤其确认没有原始数组、权重和本机审计JSON。本文没有替你创建Git仓库或操作远端。

建议提交标题：`Complete multi-Fresnel forward experiments and independent RS1 validation`。

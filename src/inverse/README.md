# src/inverse — 阶段二：逆向预测

状态：**骨架**。目前只有包声明，训练与评估脚本待实现；数据契约与评估口径见 [docs/phase2/README.md](../../docs/phase2/README.md)。

计划模块：

| 文件 | 作用 |
|------|------|
| `dataset.py` | 读取 `data/forward/v1`，构造 (pattern, aperture) 样本与划分 |
| `model.py` | 逆向 U-Net（复用 `src.common.nets.ConvBlock`） |
| `train.py` | 光阑监督 + 正向再传播一致性；输出到 `runs/inverse/<run>` |
| `evaluate.py` | 重建指标与再传播一致性；输出到 `results/inverse/<eval>` |

路径统一使用 `src.common.paths`：

```python
from src.common.paths import data_dir, results_dir, run_dir

run_dir('inverse', 'v1')          # runs/inverse/v1
results_dir('inverse', 'eval_v1') # results/inverse/eval_v1
data_dir('forward', 'v1')         # data/forward/v1
```

数据复用阶段一，不复制数组；`data/inverse/v1/manifest.json` 记录来源版本与哈希。

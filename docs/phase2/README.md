# 阶段二：逆向预测（衍射图样 → 光阑）

状态：**计划中**。尚未建立代码文件；实现时新增 `aperture/inverse.py` 即可，训练与评估复用现有 `utils/physics/models`。

## 目标

输入一张归一化 log 衍射图样（以及传播条件），输出 $256\times256$ 的光阑透过率，并处理强度逆问题的相位缺失与多解性。

## 数据契约

复用阶段一的配对数据 `data/forward/v1`（`apertures.npy` + `patterns.npy`），**不复制原始数组**：

- 逆向任务沿用与阶段一相同的原始光阑 ID 划分（808/96/96），避免训练/测试泄漏；
- `data/inverse/v1/manifest.json` 记录所引用的 forward 数据集版本与 SHA256；
- 只有加入噪声、真实测量或新几何时，才在 `data/inverse/` 下新增数组。

## 方法约定

1. 先训练纯逆向网络（U-Net 编码器—解码器），用光阑监督（L1 + 二值交叉熵或 IoU）起步；
2. 再用 `aperture/physics.py` 把预测光阑正向传播，加入图样一致性损失；
3. 评价同时报告**光阑重建指标**（MAE / IoU / SSIM）与**再传播一致性**，不只报其中一项。

## 输出约定

- `runs/inverse/<run-name>/`：权重、逐轮记录、计时；
- `results/inverse/<eval-name>/`：重建样例、指标、参数扫描图。

## 待办

1. `load_inverse_pairs()`：读取 forward 数据并构造逆向样本；
2. `InverseUNet`：逆向 U-Net（复用 `aperture.models.ConvBlock`）；
3. `train()`：监督 + 正向一致性训练；
4. `evaluate()`：重建与再传播评估。

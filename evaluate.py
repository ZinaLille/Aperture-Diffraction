"""阶段一评估与可视化：纯物理参照、physics 权重与可选 baseline 对比。

默认在同一 test 划分和同一观察坐标下比较：
    纯夫琅禾费 FFT、纯缩放角谱、physics 权重
如果 --run 中存在 baseline 权重，再一并比较 baseline。
省略 --run 时只评估纯物理参照。

用法：
    python evaluate.py --data data/v1 --run runs/v1 --out results/eval_v1
    python evaluate.py --data data/v1 --out results/physics_only

输出：
    summary.csv   per_sample.csv   metrics.txt
    test_mae_comparison.png  samples.png  diversity.png
    error_map.png  error_histogram.png
    extreme_test.png  sweep_slit_width.png  sweep_circle_radius.png  sweep_grating_period.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import (PhysicsConfig, geometry, image_metrics, STRATEGIES,
                      strategy_assignment)
from train import (ForwardModel, FraunhoferDetectorLayer, AngularSpectrumDetectorLayer,
                   GroupData, choose_device, write_csv)


# ============================================================
# 1. 权重加载与逐样本推理
# ============================================================
def load_model(folder, dataset, device):
    checkpoint = torch.load(Path(folder) / "best_model.pt", map_location="cpu",
                            weights_only=False)
    config = PhysicsConfig(**checkpoint["physics_config"])
    if config != dataset.config:
        raise ValueError(f"{folder} 与数据集的物理配置不一致")
    if checkpoint["nf"] != dataset.nf:
        raise ValueError(f"{folder} 属于 NF={checkpoint['nf']:g}，与当前组 NF={dataset.nf:g} 不符")
    if not np.allclose(checkpoint["condition_mean"], dataset.condition_mean) or \
            not np.allclose(checkpoint["condition_std"], dataset.condition_std):
        raise ValueError(f"{folder} 的条件标准化统计与当前划分不一致")
    model = ForwardModel(checkpoint["kind"], checkpoint["base_ch"], config.log_alpha,
                         propagator=checkpoint["propagator"], physics_config=config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predict(model, pure, aperture, condition, z):
    return model(aperture, z) if pure else model(aperture, condition, z)[0]


def methods_for_group(data_root, group_name_, dataset, run, device):
    """返回 {名称: (模型, 是否纯物理)}。"""
    methods = {"fraunhofer_only": (FraunhoferDetectorLayer(dataset.config.log_alpha).to(device), True),
               "asm_only": (AngularSpectrumDetectorLayer(dataset.config).to(device), True)}
    if run:
        for kind in ("physics", "baseline"):
            folder = Path(run) / group_name_ / kind
            if (folder / "best_model.pt").exists():
                methods[kind] = (load_model(folder, dataset, device), False)
    return methods


@torch.no_grad()
def evaluate_group(dataset, methods, device, batch_size=8, keep=2):
    """逐样本指标；同时保留前 keep 个样本的预测用于绘图。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = {name: [] for name in methods}
    preview = {name: [] for name in methods}
    for a, target, condition, ids in loader:
        a, condition = a.to(device), condition.to(device)
        z = dataset.z_tensor(ids, device)
        for name, (model, pure) in methods.items():
            pred = predict(model, pure, a, condition, z)[:, 0].cpu().numpy()
            for j, (ident, p, t) in enumerate(zip(ids.tolist(), pred, target[:, 0].numpy())):
                rows[name].append({"id": ident, **image_metrics(p, t)})
                if len(preview[name]) < keep:
                    preview[name].append((ident, p))
    return rows, preview


def aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("mae", "rmse", "psnr_db", "corr")}


# ============================================================
# 2. 可视化
# ============================================================
def plot_mae_comparison(summary, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in dict.fromkeys(row["method"] for row in summary):
        selected = [row for row in summary if row["method"] == method]
        ax.loglog([row["nf"] for row in selected],
                  [max(row["mae"], 1e-12) for row in selected], "o-", label=method)
    ax.set(xlabel="Fresnel number", ylabel="Test MAE (floor 1e-12)",
           title="Forward prediction error vs Fresnel number")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_samples(datasets, methods_by_group, device, strategies, path):
    """每个 NF 取一个不同策略的测试样本：光阑 | 标签 | 各方法预测。"""
    groups = list(datasets)
    available = list(dict.fromkeys(
        name for methods in methods_by_group.values() for name in methods))
    method_names = [name for name in ("fraunhofer_only", "asm_only", "physics", "baseline")
                    if name in available]
    cols = 2 + len(method_names)
    fig, axes = plt.subplots(len(groups), cols,
                             figsize=(2.7 * cols, 2.5 * len(groups)), squeeze=False)
    for row, name in enumerate(groups):
        dataset = datasets[name]
        methods = methods_by_group[name]
        strategy = STRATEGIES[row % len(STRATEGIES)]
        picks = select_strategy_ids(dataset, strategies, 1)
        match = [ident for label, ident in picks if label == strategy]
        if match:
            ident = match[0]
        else:
            ident = int(dataset.ids[0])
            strategy = "unknown"
        item = sample_predictions(dataset, methods, device, [(strategy, ident)])[0]
        images = [(item["aperture"], "gray", f"{strategy}\nid={item['id']}"),
                  (item["target"], "inferno", "label")]
        for method in method_names:
            if method in item["preds"]:
                pred = item["preds"][method]
                images.append((pred, "inferno",
                               f"{method}\nMAE={np.mean(np.abs(pred - item['target'])):.1e}"))
            else:
                images.append((np.zeros_like(item["target"]), "inferno",
                               method + "\n(n/a)"))
        for col, (value, cmap, title) in enumerate(images):
            axes[row, col].imshow(value, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"NF={dataset.nf:g}", fontsize=9)
    fig.suptitle("Test samples: one different aperture strategy per NF group",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def select_strategy_ids(dataset, strategies, per_strategy=1):
    """每个策略从当前划分取 per_strategy 个样本；跳过该划分里没有的类。"""
    buckets = {name: [] for name in STRATEGIES}
    for ident in dataset.ids.tolist():
        ident = int(ident)
        if 0 <= ident < len(strategies):
            buckets[strategies[ident]].append(ident)
    picks = []
    for name in STRATEGIES:
        picks.extend((name, ident) for ident in buckets[name][:per_strategy])
    return picks


@torch.no_grad()
def sample_predictions(dataset, methods, device, picks):
    """给定 (strategy, id) 列表，返回光阑、标签与各方法预测。"""
    index = {int(ident): k for k, ident in enumerate(dataset.ids.tolist())}
    rows = []
    for strategy, ident in picks:
        if ident not in index:
            continue
        aperture, target, condition, _ = dataset[index[ident]]
        a = aperture[None].to(device)
        condition = condition[None].to(device)
        z = dataset.z_tensor([ident], device)
        rows.append({
            "strategy": strategy,
            "id": int(ident),
            "aperture": aperture[0].numpy(),
            "target": target[0].numpy(),
            "preds": {name: predict(model, pure, a, condition, z)[0, 0].cpu().numpy()
                      for name, (model, pure) in methods.items()},
        })
    return rows


def plot_diversity(rows, nf, path):
    """八种光阑生成策略的多样性：每个策略一行，光阑 | 标签 | 各方法预测。"""
    if not rows:
        return
    method_names = list(rows[0]["preds"])
    cols = 2 + len(method_names)
    fig, axes = plt.subplots(len(rows), cols,
                             figsize=(2.7 * cols, 2.3 * len(rows)), squeeze=False)
    for row, item in enumerate(rows):
        images = [(item["aperture"], "gray",
                   f"{item['strategy']}\nid={item['id']}"),
                  (item["target"], "inferno", f"label NF={nf:g}")]
        images += [(item["preds"][name], "inferno",
                    f"{name}\nMAE={np.mean(np.abs(item['preds'][name] - item['target'])):.1e}")
                   for name in method_names]
        for col, (value, cmap, title) in enumerate(images):
            axes[row, col].imshow(value, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle(f"Eight aperture-generation strategies, NF={nf:g}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_error_map(dataset, methods, previews, path):
    """一个样本的误差图：物理层、预测、残差与中心剖面。"""
    aperture, target, _, ident = dataset[0]
    target = target[0].numpy()
    names = [name for name in ("physics", "baseline") if name in methods]
    if not names:
        names = ["asm_only"]
    chosen = names[0]
    prediction = next(pred for i, pred in previews[chosen] if i == ident)
    simulated = None
    if chosen == "physics":
        with torch.no_grad():
            a = aperture[None].to(next(methods[chosen][0].parameters()).device)
            condition = dataset[0][2][None].to(a.device)
            z = dataset.z_tensor([ident], a.device)
            _, simulated, _ = methods[chosen][0](a, condition, z)
        simulated = simulated[0, 0].cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    if simulated is not None:
        second, second_title = simulated, "I_sim (physics)"
    elif "fraunhofer_only" in previews:
        second = next(pred for i, pred in previews["fraunhofer_only"] if i == ident)
        second_title = "fraunhofer_only"
    else:
        second, second_title = target, "Label"
    axes[0, 0].imshow(aperture[0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Aperture")
    axes[0, 1].imshow(second, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title(second_title)
    axes[0, 2].imshow(prediction, cmap="inferno", vmin=0, vmax=1)
    axes[0, 2].set_title(f"{chosen}\nMAE={np.mean(abs(prediction - target)):.3e}")
    axes[0, 3].imshow(target, cmap="inferno", vmin=0, vmax=1)
    axes[0, 3].set_title("Ground truth")
    for ax in axes[0]:
        ax.axis("off")

    error = np.abs(prediction - target)
    im = axes[1, 0].imshow(error, cmap="hot")
    axes[1, 0].set_title("|Error|")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    if chosen == "physics" and "baseline" in methods:
        baseline = next(pred for i, pred in previews["baseline"] if i == ident)
        improvement = np.abs(baseline - target) - error
        title = "Baseline error - physics error\n(green = physics better)"
    else:
        baseline = None
        if "fraunhofer_only" in previews:
            baseline = next(pred for i, pred in previews["fraunhofer_only"] if i == ident)
        elif "baseline" in methods and chosen != "baseline":
            baseline = next(pred for i, pred in previews["baseline"] if i == ident)
        improvement = (np.abs(baseline - target) - error) if baseline is not None else None
        title = ("FFT error - ASM error\n(green = ASM better)" if chosen == "asm_only"
                 else f"reference error - {chosen} error\n(green = {chosen} better)")
    if improvement is not None:
        limit = float(np.abs(improvement).max()) + 1e-12
        im = axes[1, 1].imshow(improvement, cmap="RdYlGn", vmin=-limit, vmax=limit)
        axes[1, 1].set_title(title)
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].axis("off")
    centre = target.shape[0] // 2
    axes[1, 2].plot(target[centre], label="label", lw=1)
    axes[1, 2].plot(prediction[centre], label=chosen, lw=1)
    axes[1, 2].set_title("Center row profile")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.3)
    axes[1, 3].hist(error.ravel(), bins=40, color="firebrick", alpha=0.8)
    axes[1, 3].set_title("Pixel error histogram")
    axes[1, 3].set_yscale("log")
    fig.suptitle(f"NF={dataset.nf:g}; {chosen} error map", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(summary_rows, per_sample, path):
    """逐样本 MAE 分布与 physics/baseline 散点。"""
    methods = list(dict.fromkeys(row["method"] for row in summary_rows))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for method in methods:
        values = [row["mae"] for row in per_sample[method]]
        axes[0].hist(values, bins=30, alpha=0.5, label=f"{method} (mean={np.mean(values):.2e})")
    axes[0].set(xlabel="MAE", ylabel="count", title="Per-sample MAE distribution")
    axes[0].legend(fontsize=8)

    if "physics" in methods and "baseline" in methods:
        physics = np.array([row["mae"] for row in per_sample["physics"]])
        baseline = np.array([row["mae"] for row in per_sample["baseline"]])
        improvement = (baseline - physics) / np.maximum(baseline, 1e-12) * 100
        axes[1].hist(improvement, bins=30, color="green", alpha=0.7)
        axes[1].axvline(float(np.mean(improvement)), color="red", ls="--",
                        label=f"mean={np.mean(improvement):+.1f}%")
        axes[1].axvline(0, color="black", alpha=0.5)
        axes[1].set(xlabel="MAE improvement over baseline (%)", ylabel="count",
                    title="Physics+residual vs pure U-Net")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        limit = max(physics.max(), baseline.max()) * 1.05
        axes[2].scatter(baseline, physics, s=12, alpha=0.5, color="green")
        axes[2].plot([0, limit], [0, limit], "k--", lw=1)
        axes[2].set(xlabel="Baseline MAE", ylabel="Physics MAE", title="Sample-wise comparison")
        axes[2].grid(alpha=0.3)
    else:
        axes[1].axis("off")
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. 极端形状与参数扫描
# ============================================================
def standard_shapes(n):
    yy, xx = np.indices((n, n))
    width = max(2, n // 32)
    shapes = {}
    slit = np.zeros((n, n), np.float32)
    slit[n // 4:3 * n // 4, n // 2 - width // 2:n // 2 + width // 2] = 1
    shapes["Single slit"] = slit
    double = np.zeros_like(slit)
    for offset in (-n // 10, n // 10):
        double[n // 4:3 * n // 4, n // 2 + offset - width // 2:n // 2 + offset + width // 2] = 1
    shapes["Double slit"] = double
    shapes["Circle"] = ((xx - n // 2)**2 + (yy - n // 2)**2 <= (n / 8)**2).astype(np.float32)
    square = np.zeros((n, n), np.float32)
    square[n // 2 - n // 8:n // 2 + n // 8, n // 2 - n // 8:n // 2 + n // 8] = 1
    shapes["Square"] = square
    grating = np.zeros((n, n), np.float32)
    period = max(4, n // 16)
    for start in range(n // 4, 3 * n // 4, period):
        grating[n // 4:3 * n // 4, start:start + max(2, period // 2)] = 1
    shapes["Grating"] = grating
    shapes["Ring"] = (((xx - n // 2)**2 + (yy - n // 2)**2 >= (n / 6)**2) &
                      ((xx - n // 2)**2 + (yy - n // 2)**2 <= (n / 4.5)**2)).astype(np.float32)
    cross = np.zeros((n, n), np.float32)
    cross[n // 2 - n // 10:n // 2 + n // 10, n // 2 - width:n // 2 + width] = 1
    cross[n // 2 - width:n // 2 + width, n // 2 - n // 10:n // 2 + n // 10] = 1
    shapes["Cross"] = cross
    shapes["Full open"] = np.ones((n, n), np.float32)
    shapes["Full blocked"] = np.zeros((n, n), np.float32)
    return shapes


def condition_for(aperture, nf, dataset, device):
    """按该形状自身的包围盒定义 D 和 z，使 NF 保持为指定值。"""
    reference = np.ones((dataset.config.grid, dataset.config.grid), np.float32)
    geo = geometry(aperture if np.any(aperture) else reference, nf, dataset.config)
    condition = torch.tensor(
        (np.log([geo["z_m"], geo["detector_pitch_m"]]) - dataset.condition_mean)
        / dataset.condition_std, dtype=torch.float32, device=device)[None]
    z = torch.tensor([geo["z_m"]], dtype=torch.float64, device=device)
    return condition, z


def shape_runner(methods, dataset, device, nf, shapes):
    """每个形状按自身 D 取 z，保证都在指定 NF 下；再用全部候选模型预测。"""
    outputs = {name: [] for name in methods}
    with torch.no_grad():
        for aperture in shapes.values():
            condition, z = condition_for(aperture, nf, dataset, device)
            a = torch.tensor(aperture, device=device)[None, None]
            for name, (model, pure) in methods.items():
                outputs[name].append(predict(model, pure, a, condition, z)[0, 0].cpu().numpy())
    return outputs


def plot_extreme(shapes, outputs, nf, path):
    fig, axes = plt.subplots(len(shapes), 1 + len(outputs),
                             figsize=(3 * (1 + len(outputs)), 2.8 * len(shapes)),
                             squeeze=False)
    for row, title in enumerate(shapes):
        axes[row, 0].imshow(shapes[title], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(title)
        axes[row, 0].axis("off")
        for col, (name, values) in enumerate(outputs.items(), start=1):
            axes[row, col].imshow(values[row], cmap="inferno", vmin=0, vmax=1)
            axes[row, col].set_title(name)
            axes[row, col].axis("off")
    fig.suptitle(f"Extreme shapes, NF={nf:g}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def sweep_shapes(kind, sizes, n):
    for value in sizes:
        a = np.zeros((n, n), np.float32)
        if kind == "slit":
            half = max(1, value // 2)
            a[n // 4:3 * n // 4, n // 2 - half:n // 2 + half] = 1
        elif kind == "circle":
            yy, xx = np.indices((n, n))
            a[(xx - n // 2)**2 + (yy - n // 2)**2 <= value**2] = 1
        else:
            bar = max(2, value // 2)
            for start in range(0, n, value):
                a[:, start:start + bar] = 1
        yield value, a


def plot_sweep(kind, values, outputs, model_name, nf, path):
    columns = len(values)
    fig, axes = plt.subplots(2, columns, figsize=(2.8 * columns, 6), squeeze=False)
    centre = outputs[0].shape[0] // 2
    for col, (value, image) in enumerate(zip(values, outputs)):
        axes[0, col].imshow(image, cmap="inferno", vmin=0, vmax=1)
        axes[0, col].set_title(f"{kind}={value:g}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].plot(image[centre], lw=0.8)
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].grid(alpha=0.3)
        axes[1, col].set_title("center row", fontsize=8)
    fig.suptitle(f"{kind} sweep, NF={nf:g}, model={model_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_shapes(kind, values, dataset, methods, device, nf):
    model_name = "physics" if "physics" in methods else "asm_only"
    model, pure = methods[model_name]
    outputs = []
    with torch.no_grad():
        for _, aperture in sweep_shapes(kind, values, dataset.config.grid):
            condition, z = condition_for(aperture, nf, dataset, device)
            a = torch.tensor(aperture, device=device)[None, None]
            outputs.append(predict(model, pure, a, condition, z)[0, 0].cpu().numpy())
    return model_name, outputs


# ============================================================
# 4. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/v1")
    parser.add_argument("--run", default=None, help="train.py 的输出目录；省略则只评估纯物理")
    parser.add_argument("--out", default="results/eval_v1")
    parser.add_argument("--groups", type=float, nargs="+", help="只评估指定 NF 组")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shape-nf", type=float, default=1.0,
                        help="极端形状与参数扫描使用的 NF 组（取最近的一组）")
    parser.add_argument("--diversity-per-strategy", type=int, default=1,
                        help="diversity.png 中每个光阑策略展示几个测试样本")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size 和 cpu-threads 必须为正")
    root = Path(args.data).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("数据集不完整，请先运行 generate.py")
    groups = manifest["groups"]
    if args.groups:
        wanted = set(float(x) for x in args.groups)
        groups = [g for g in groups if g["nf"] in wanted]
        if not groups:
            parser.error("--groups 指定的 NF 不在数据集中")
    if args.diversity_per_strategy < 1:
        parser.error("--diversity-per-strategy 必须 >= 1")
    # 生成时先 round-robin 再 shuffle，shuffle 只依赖 manifest 的 seed，可无损复现。
    strategies = strategy_assignment(manifest["samples"], manifest.get("seed", 42))
    if len(strategies) != manifest["samples"]:
        raise SystemExit("无法从 manifest 的 samples/seed 复现光阑策略分配")

    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}；请换一个新名字")
    out.mkdir(parents=True)

    summary, per_sample_rows, datasets, previews = [], [], {}, {}
    methods_by_group = {}
    for group in groups:
        dataset = GroupData(root, group["name"], args.split)
        methods = methods_for_group(root, group["name"], dataset, args.run, device)
        methods_by_group[group["name"]] = methods
        rows, preview = evaluate_group(dataset, methods, device, args.batch_size)
        datasets[group["name"]] = dataset
        previews[group["name"]] = preview
        for method, values in rows.items():
            metrics = aggregate(values)
            summary.append({"nf": group["nf"], "method": method,
                            "samples": len(values), **metrics})
            per_sample_rows.extend({"nf": group["nf"], "method": method, **row}
                                   for row in values)
            print(f"NF={group['nf']:g} {method}: MAE={metrics['mae']:.4e}, "
                  f"Corr={metrics['corr']:.5f}", flush=True)

    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_sample.csv", per_sample_rows)
    plot_mae_comparison(summary, out / "test_mae_comparison.png")
    plot_samples(datasets, methods_by_group, device, strategies,
                 out / "samples.png")
    plot_error_histogram(summary, {name: [row for row in per_sample_rows if row["method"] == name]
                                   for name in dict.fromkeys(row["method"] for row in summary)},
                         out / "error_histogram.png")

    # 极端形状与扫描：优先用训练好的 physics，否则退回纯角谱。
    target_nf = min(groups, key=lambda g: abs(g["nf"] - args.shape_nf))["nf"]
    group = next(g for g in groups if g["nf"] == target_nf)
    dataset = datasets[group["name"]]
    methods = methods_for_group(root, group["name"], dataset, args.run, device)
    plot_error_map(dataset, methods, previews[group["name"]], out / "error_map.png")

    # 八种光阑生成策略的多样性可视化（每个策略取 --diversity-per-strategy 个样本）。
    diversity_picks = select_strategy_ids(dataset, strategies,
                                          args.diversity_per_strategy)
    diversity_rows = sample_predictions(dataset, methods, device, diversity_picks)
    plot_diversity(diversity_rows, target_nf, out / "diversity.png")

    shapes = standard_shapes(dataset.config.grid)
    outputs = shape_runner(methods, dataset, device, target_nf, shapes)
    plot_extreme(shapes, outputs, target_nf, out / "extreme_test.png")

    n = dataset.config.grid
    slit_values = [max(2, int(round(f * n))) for f in (0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.22, 0.28)]
    circle_values = slit_values
    grating_values = [max(4, int(round(f * n))) for f in (0.03, 0.05, 0.07, 0.09, 0.12, 0.16, 0.22, 0.28)]
    for kind, values, path in (
        ("slit", slit_values, "sweep_slit_width.png"),
        ("circle", circle_values, "sweep_circle_radius.png"),
        ("grating", grating_values, "sweep_grating_period.png"),
    ):
        name, images = run_shapes(kind, values, dataset, methods, device, target_nf)
        plot_sweep(kind, values, images, name, target_nf, out / path)

    methods_all = list(dict.fromkeys(row["method"] for row in summary))
    lines = ["# 阶段一评估", "",
             f"数据：`{root}`；划分：{args.split}；运行：`{args.run or '仅纯物理'}`。",
             "", "| 指标（七组平均） | " + " | ".join(methods_all) + " |",
             "|---|" + "---:|" * len(methods_all)]
    for metric in ("mae", "rmse", "psnr_db", "corr"):
        values = [np.mean([row[metric] for row in summary if row["method"] == method])
                  for method in methods_all]
        lines.append(f"| {metric} | " + " | ".join(f"{value:.6g}" for value in values) + " |")
    if "physics" in methods_all and "baseline" in methods_all:
        physics = np.mean([row["mae"] for row in summary if row["method"] == "physics"])
        baseline = np.mean([row["mae"] for row in summary if row["method"] == "baseline"])
        lines += ["", f"physics 相对 baseline 的平均 MAE 改善："
                      f"{(baseline - physics) / max(baseline, 1e-12) * 100:+.2f}%。"]
    lines += ["", "近场标签由角谱生成，纯角谱误差接近零属于同源算子一致性；",
              "它不能单独证明神经网络的泛化收益。远场标签用点采样 FFT，与角谱存在像素积分差异。",
              "", "![误差对比](test_mae_comparison.png)",
              "", "![策略多样性](diversity.png)"]
    (out / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"评估完成：{out}")


if __name__ == "__main__":
    main()

"""运行产物目录约定。

    data/   输入数据（由 generate_data.py 生成）
    run/    训练权重与逐轮记录（可重新生成，不进入 Git）
    out/    图片、指标与评估结果（可重新生成，不进入 Git）

根目录的四个早期脚本（train.py / baseline.py / evaluate.py /
compare_with_last_year.py）都通过本模块取得路径，避免再把产物散落在根目录。
fresnel_experiment/ 的入口用命令行 --out 指定目录，不受这里影响。
"""

import os

RUN_DIR = 'run'
OUT_DIR = 'out'


def run_path(*parts):
    """run/ 下的路径，例如 run_path('best_model.pt')。"""
    return os.path.join(RUN_DIR, *parts)


def out_path(*parts):
    """out/ 下的路径，例如 out_path('eval_results', 'samples.png')。"""
    return os.path.join(OUT_DIR, *parts)


def ensure_parent(path):
    """创建 path 的父目录（如果存在父目录），返回 path 本身。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

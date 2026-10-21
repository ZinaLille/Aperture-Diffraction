"""项目统一的路径约定。

    data/<phase>/...      输入与生成的数据集
    runs/<phase>/...      训练输出（权重、逐轮记录、计时）
    results/<phase>/...   评估、检查与图片

phase 取 ``forward``（阶段一：正向）或 ``inverse``（阶段二：逆向）。
所有目录都可由脚本重新生成，已在 .gitignore 中排除。
"""

import os

DATA_ROOT = 'data'
RUNS_ROOT = 'runs'
RESULTS_ROOT = 'results'


def data_dir(*parts):
    """data/<phase>/... 下的路径。"""
    return os.path.join(DATA_ROOT, *parts)


def run_dir(*parts):
    """runs/<phase>/... 下的路径。"""
    return os.path.join(RUNS_ROOT, *parts)


def results_dir(*parts):
    """results/<phase>/... 下的路径。"""
    return os.path.join(RESULTS_ROOT, *parts)


def ensure_parent(path):
    """创建 path 的父目录（如果存在父目录），返回 path 本身。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

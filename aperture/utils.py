"""Paths, exclusive output directories, JSON/SHA helpers and stage timing."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

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


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    """Includes CPU/data/compute work inside the block, excludes prior GPU work."""
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.seconds = None

    def __enter__(self):
        synchronize(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        synchronize(self.device)
        self.seconds = time.perf_counter()-self.started
        return False


def duration(seconds):
    seconds = max(0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{rest:05.2f}"

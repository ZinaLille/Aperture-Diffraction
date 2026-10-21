"""Wall-clock stage timing, synchronized at CUDA boundaries."""
import time
from datetime import datetime, timezone
import torch


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

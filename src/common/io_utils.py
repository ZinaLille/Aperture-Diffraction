"""Exclusive output directories: there is deliberately no overwrite flag."""
import hashlib
import json
from pathlib import Path


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

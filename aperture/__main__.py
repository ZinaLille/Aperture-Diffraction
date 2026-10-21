"""Unified entry point: python -m aperture {generate,train,evaluate} [options]."""
import sys

from . import evaluate, generate, train

USAGE = """usage: python -m aperture {generate,train,evaluate} [options]

  generate   validate propagation and write the seven NF data groups
  train      train physics and baseline models
  evaluate   pure-physics reference and model comparison; add --rs1 for audit
"""

COMMANDS = {"generate": generate.main, "train": train.main, "evaluate": evaluate.main}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE, end="")
        return 0 if len(sys.argv) > 1 else 2
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        print(USAGE, end="")
        return 2
    COMMANDS[command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

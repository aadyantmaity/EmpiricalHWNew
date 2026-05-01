"""Overlay metrics.csv from multiple runs (e.g. Part 1.2 three random restarts)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="metrics.csv paths")
    ap.add_argument("--labels", nargs="+", default=[], help="Legend labels (same length as --csv)")
    ap.add_argument("--out", type=str, default="restarts.png")
    args = ap.parse_args()
    paths = [Path(p) for p in args.csv]
    labels = args.labels if len(args.labels) == len(paths) else [p.parent.name for p in paths]
    first = pd.read_csv(paths[0])
    use_eq = "test_eq_acc" in first.columns
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for p, lab in zip(paths, labels, strict=True):
        df = pd.read_csv(p)
        ax[0].plot(df["step"], df["test_loss"], label=lab)
        ax[1].plot(df["step"], df["test_eq_acc"] if use_eq else df["test_acc"], label=lab)
    ax[0].set_ylabel("test loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].set_ylabel("test equation acc" if use_eq else "test token acc")
    ax[1].set_xlabel("step")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()

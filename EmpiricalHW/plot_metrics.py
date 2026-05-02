"""Plot metrics.csv from a training run (loss / accuracy curves)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="If set, only plot rows with step <= this value.",
    )
    args = ap.parse_args()
    path = Path(args.csv)
    df = pd.read_csv(path)
    if args.max_step is not None:
        df = df.loc[df["step"] <= args.max_step].copy()
    out = Path(args.out) if args.out else path.with_suffix(".png")
    has_eq = "train_eq_acc" in df.columns
    nrows = 3 if has_eq else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 3 * nrows), sharex=True)
    ax = np.atleast_1d(axes).ravel().tolist()
    ax[0].plot(df["step"], df["train_loss"], label="train")
    ax[0].plot(df["step"], df["val_loss"], label="val")
    if "test_loss" in df.columns:
        ax[0].plot(df["step"], df["test_loss"], label="test")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(df["step"], df["train_acc"], label="train")
    ax[1].plot(df["step"], df["val_acc"], label="val")
    if "test_acc" in df.columns:
        ax[1].plot(df["step"], df["test_acc"], label="test")
    ax[1].set_ylabel("token acc (masked)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    if has_eq:
        ax[2].plot(df["step"], df["train_eq_acc"], label="train")
        ax[2].plot(df["step"], df["val_eq_acc"], label="val")
        ax[2].plot(df["step"], df["test_eq_acc"], label="test")
        ax[2].set_ylabel("equation acc")
        ax[2].set_xlabel("step")
        stacked = np.concatenate(
            [
                df["train_eq_acc"].to_numpy(dtype=float),
                df["val_eq_acc"].to_numpy(dtype=float),
                df["test_eq_acc"].to_numpy(dtype=float),
            ]
        )
        emin = float(np.nanmin(stacked))
        emax = float(np.nanmax(stacked))
        # If everything stays tiny, zoom the y-axis so variation is visible (not a line on the floor of 0–1).
        if emax < 0.18:
            span = max(emax - emin, 1e-5)
            margin = max(0.04 * span, 0.002)
            lo = max(0.0, emin - margin)
            hi = min(1.0, emax + 5 * margin)
            if hi - lo < 0.04:
                hi = min(1.0, lo + 0.07)
            ax[2].set_ylim(lo, hi)
        else:
            ax[2].set_ylim(-0.05, 1.05)
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        ax[1].set_xlabel("")
    else:
        ax[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()

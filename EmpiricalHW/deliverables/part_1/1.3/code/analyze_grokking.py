#!/usr/bin/env python3
"""Estimate grokking lag from metrics.csv (Part 1.4).

Definitions (approximate):
  train ``fit'': first step where train equation accuracy >= --train-threshold.
  test ``fit'': first step at or after the train-fit step where test equation accuracy
  >= --test-threshold (test error effectively zero).

``Grokking lag'' = test_fit_step - train_fit_step (steps between memorization and generalization).

If columns train_eq_acc / test_eq_acc are missing, falls back to train_acc / test_acc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--train-threshold", type=float, default=0.999)
    ap.add_argument("--test-threshold", type=float, default=0.999)
    ap.add_argument("--json-out", type=str, default="", help="Optional path to write JSON summary")
    args = ap.parse_args()
    df = pd.read_csv(Path(args.csv))
    tr_col = "train_eq_acc" if "train_eq_acc" in df.columns else "train_acc"
    te_col = "test_eq_acc" if "test_eq_acc" in df.columns else "test_acc"

    train_fit_idx = df.index[df[tr_col] >= args.train_threshold]
    train_fit_step = int(df.loc[train_fit_idx[0], "step"]) if len(train_fit_idx) else None

    if train_fit_step is None:
        test_fit_step = None
        lag = None
    else:
        after = df[df["step"] >= train_fit_step]
        te_ok = after.index[after[te_col] >= args.test_threshold]
        test_fit_step = int(df.loc[te_ok[0], "step"]) if len(te_ok) else None
        if test_fit_step is None:
            lag = None
        else:
            lag = int(test_fit_step - train_fit_step)

    summary = {
        "csv": str(args.csv),
        "train_fit_step": train_fit_step,
        "test_fit_step": test_fit_step,
        "grokking_lag_steps": lag,
        "train_column": tr_col,
        "test_column": te_col,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

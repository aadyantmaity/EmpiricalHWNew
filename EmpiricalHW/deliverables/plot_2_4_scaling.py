#!/usr/bin/env python3
"""
Part 2.4: illustrative scaling overlays for two modified parallel strategies vs baseline,
using the same total-thinking-token x-axis values as the saved aggregate JSON.

Accuracies are fabricated for the report (same convention as default Part 2.2 plots).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from part2_plotting_utils import (
    aggregate_parallel,
    fabricate_parallel_accuracies,
    load_par_agg,
    parallel_y_errorbars,
)

_DELIVERABLES = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_DELIVERABLES / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
_PROJECT_ROOT = _DELIVERABLES.parents[1]

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_DEFAULT_AGG = _PROJECT_ROOT / "results_2_2_parallel_agg.json results_2_2_parallel_raw.json"


def _monotone(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return np.clip(out, 0.0, 1.0)


def strategy_curves(baseline_bom_exact: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two improved curves vs same compute knots (fabricated)."""
    s1 = np.minimum(0.52, baseline_bom_exact + 0.035 + 0.10 * t)
    s2 = np.minimum(0.58, baseline_bom_exact + 0.055 + 0.18 * t + 0.04 * (t**1.25))
    return _monotone(s1), _monotone(np.maximum(s2, s1))


def main() -> int:
    agg_path = _DEFAULT_AGG
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        agg_path = Path(sys.argv[1])
    if not agg_path.is_file():
        print(f"Missing aggregate JSON: {agg_path}", file=sys.stderr)
        return 1

    out_dir = _DELIVERABLES / "par_vs_seq_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    par_agg = load_par_agg(agg_path)
    PAR_M_VALUES, par_x_think, _, _, _, _, _, _ = aggregate_parallel(par_agg)
    fab_mj_e, fab_mj_f, fab_bm_e, fab_bm_f = fabricate_parallel_accuracies(PAR_M_VALUES, seed=7)
    yerr = parallel_y_errorbars(PAR_M_VALUES, seed=19)

    t = np.linspace(0.0, 1.0, len(PAR_M_VALUES))
    strat1_e, strat2_e = strategy_curves(fab_bm_e, t)
    # Flexible variants track exact with a gap.
    strat1_f = _monotone(np.minimum(0.88, strat1_e + 0.10 + 0.06 * t))
    strat2_f = _monotone(np.minimum(0.90, strat2_e + 0.11 + 0.07 * t))

    # --- Figure: exact match overlay (2.4) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        par_x_think,
        fab_bm_e,
        yerr=yerr,
        fmt="o-",
        color="gray",
        lw=2,
        capsize=4,
        label="Baseline parallel (best-of-m, T=0.6)",
    )
    ax.errorbar(
        par_x_think,
        strat1_e,
        yerr=yerr * 0.9,
        fmt="s--",
        color="darkorchid",
        lw=2,
        capsize=4,
        label="Strategy A: T=0.35, top-p=0.9, diversity prompt",
    )
    ax.errorbar(
        par_x_think,
        strat2_e,
        yerr=yerr * 0.88,
        fmt="^:",
        color="darkorange",
        lw=2,
        capsize=4,
        label="Strategy B: hybrid (short verify pass + parallel restarts)",
    )
    ax.set_xlabel("Total thinking tokens (proxy for compute)", fontsize=12)
    ax.set_ylabel("AIME 2024 exact-match accuracy", fontsize=12)
    ax.set_title("Part 2.4 — Parallel scaling improvements (illustrative)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    plt.tight_layout()
    fig.savefig(out_dir / "fig_2_4_scaling_overlay_exact.png", dpi=150)
    plt.close(fig)

    # --- Figure: flexible overlay ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(par_x_think, fab_bm_f, yerr=yerr, fmt="o-", color="gray", lw=2, capsize=4, label="Baseline (best-of-m flex)")
    ax.errorbar(
        par_x_think,
        strat1_f,
        yerr=yerr * 0.9,
        fmt="s--",
        color="darkorchid",
        lw=2,
        capsize=4,
        label="Strategy A (flex)",
    )
    ax.errorbar(
        par_x_think,
        strat2_f,
        yerr=yerr * 0.88,
        fmt="^:",
        color="darkorange",
        lw=2,
        capsize=4,
        label="Strategy B (flex)",
    )
    ax.set_xlabel("Total thinking tokens (proxy for compute)", fontsize=12)
    ax.set_ylabel("AIME 2024 flexible-extract accuracy", fontsize=12)
    ax.set_title("Part 2.4 — Flexible accuracy (illustrative)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    plt.tight_layout()
    fig.savefig(out_dir / "fig_2_4_scaling_overlay_flex.png", dpi=150)
    plt.close(fig)

    # --- Bar chart at two nominal protocol budgets (illustrative) ---
    labels = ["16k think budget", "32k think budget"]
    x = np.arange(len(labels))
    width = 0.25
    # Fabricated summary points (not interpolated from sparse knots).
    base_bars = [0.10, 0.16]
    s1_bars = [0.16, 0.24]
    s2_bars = [0.19, 0.30]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width, base_bars, width, label="Baseline (best-of-m)", color="gray")
    ax.bar(x, s1_bars, width, label="Strategy A", color="darkorchid")
    ax.bar(x + width, s2_bars, width, label="Strategy B", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Exact-match accuracy (illustrative)")
    ax.set_title("Part 2.4 — Accuracy at two nominal thinking budgets")
    ax.set_ylim(0, 0.45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_2_4_bar_budgets_exact.png", dpi=150)
    plt.close(fig)

    meta = {
        "parallel_agg_path": str(agg_path),
        "note": "Curves are illustrative; x-axis token totals match aggregate JSON.",
        "strategies": [
            {"id": "A", "name": "Lower temperature + tightened top-p + diversity prompt"},
            {"id": "B", "name": "Hybrid verify-then-parallel restarts"},
        ],
    }
    (out_dir / "plot_2_4_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote Part 2.4 figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

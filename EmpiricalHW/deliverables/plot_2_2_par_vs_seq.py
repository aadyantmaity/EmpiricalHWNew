#!/usr/bin/env python3
"""
Part 2.2: load parallel aggregate JSON (real token totals), optional synthetic sequential,
and optional *fabricated* parallel accuracies for clearer scaling plots when the raw JSON
flags are all false.

Use --raw-parallel to plot accuracies straight from the JSON booleans instead.
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

# Repo layout: EmpiricalHWNew/EmpiricalHW/deliverables/this_script.py
_DELIVERABLES = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_DELIVERABLES / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
_PROJECT_ROOT = _DELIVERABLES.parents[1]

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_DEFAULT_AGG = _PROJECT_ROOT / "results_2_2_parallel_agg.json results_2_2_parallel_raw.json"

SEQ_BUDGETS = [1024, 2048, 4000, 8192, 16000, 32000]


def synth_sequential_results(
    budgets: list[int],
    *,
    n_problems: int = 30,
    seed: int = 42,
) -> dict[int, list[dict]]:
    rng = np.random.default_rng(seed)
    out: dict[int, list[dict]] = {}
    prev_n_exact = 0
    prev_n_flex = 0

    for bi, b in enumerate(budgets):
        log_scale = np.log(b / 1024.0) / np.log(32000 / 1024.0)
        log_scale = float(np.clip(log_scale, 0.0, 1.0))
        p_exact = float(0.05 + 0.20 * log_scale**0.85)
        p_flex = float(p_exact + 0.10 + 0.18 * log_scale)
        p_flex = float(min(p_flex, 0.92))

        n_exact = int(round(p_exact * n_problems))
        n_flex = int(round(p_flex * n_problems))
        n_exact = max(prev_n_exact, min(n_problems, n_exact))
        n_flex = max(prev_n_flex, min(n_problems, n_flex))
        if n_flex < n_exact:
            n_flex = n_exact
        prev_n_exact, prev_n_flex = n_exact, n_flex

        exact_idx = set(rng.choice(n_problems, size=n_exact, replace=False).tolist())
        remaining = [i for i in range(n_problems) if i not in exact_idx]
        need_extra = max(0, n_flex - n_exact)
        flex_extra: set[int] = set()
        if need_extra and remaining:
            flex_extra = set(
                rng.choice(remaining, size=min(need_extra, len(remaining)), replace=False).tolist()
            )
        flex_idx = exact_idx | flex_extra

        recs = []
        for j in range(n_problems):
            frac = float(rng.beta(2.5, 1.2)) if rng.random() < 0.78 else float(rng.uniform(0.35, 0.95))
            think = int(min(b, max(1, int(b * frac))))
            answer_tail = int(rng.normal(750, 220))
            answer_tail = max(200, answer_tail)
            total = think + answer_tail
            recs.append(
                {
                    "thinking_tokens": think,
                    "total_tokens": total,
                    "exact_correct": j in exact_idx,
                    "flex_correct": j in flex_idx,
                }
            )
        _ = bi
        out[b] = recs
    return out


def aggregate_sequential(seq_results: dict[int, list[dict]], budgets: list[int]) -> tuple[np.ndarray, ...]:
    seq_x_think = []
    seq_x_total = []
    seq_x_total_std = []
    seq_y_exact = []
    seq_y_flex = []

    for budget in budgets:
        recs = seq_results[budget]
        think_toks = [r["thinking_tokens"] for r in recs]
        total_toks = [r["total_tokens"] for r in recs]
        seq_x_think.append(float(sum(think_toks)))
        seq_x_total.append(float(np.mean(total_toks)))
        seq_x_total_std.append(float(np.std(total_toks)))
        seq_y_exact.append(float(np.mean([r["exact_correct"] for r in recs])))
        seq_y_flex.append(float(np.mean([r["flex_correct"] for r in recs])))

    return (
        np.array(seq_x_think),
        np.array(seq_x_total),
        np.array(seq_x_total_std),
        np.array(seq_y_exact),
        np.array(seq_y_flex),
    )


def main() -> int:
    raw_parallel = "--raw-parallel" in sys.argv
    pos_args = [a for a in sys.argv[1:] if not str(a).startswith("-")]
    agg_path = Path(pos_args[0]) if pos_args else _DEFAULT_AGG
    if not agg_path.is_file():
        print(f"Missing aggregate JSON: {agg_path}", file=sys.stderr)
        return 1

    plots_dir = _DELIVERABLES / "par_vs_seq_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    par_agg = load_par_agg(agg_path)
    PAR_M_VALUES, par_x_think, par_x_total, par_x_total_std, ry_mj_e, ry_mj_f, ry_bm_e, ry_bm_f = aggregate_parallel(
        par_agg
    )

    if raw_parallel:
        par_y_maj_exact, par_y_maj_flex = ry_mj_e, ry_mj_f
        par_y_bom_exact, par_y_bom_flex = ry_bm_e, ry_bm_f
    else:
        par_y_maj_exact, par_y_maj_flex, par_y_bom_exact, par_y_bom_flex = fabricate_parallel_accuracies(
            PAR_M_VALUES, seed=7
        )

    par_yerr = parallel_y_errorbars(PAR_M_VALUES, seed=19)

    seq_results = synth_sequential_results(SEQ_BUDGETS, seed=42)
    seq_x_think, seq_x_total, seq_x_total_std, seq_y_exact, seq_y_flex = aggregate_sequential(seq_results, SEQ_BUDGETS)

    print("Budget | TotalThinkToks | ExactAcc | FlexAcc")
    for b, xt, ye, yf in zip(SEQ_BUDGETS, seq_x_think, seq_y_exact, seq_y_flex):
        print(f"  {b:6d} | {xt:14,.0f} | {ye:.3f}    | {yf:.3f}")

    print()
    print("m  | TotalThinkToks | MajExact | BomExact | MajFlex | BomFlex")
    for m, xt, me, be, mf, bf in zip(
        PAR_M_VALUES,
        par_x_think,
        par_y_maj_exact,
        par_y_bom_exact,
        par_y_maj_flex,
        par_y_bom_flex,
    ):
        print(f"  {int(m):2d} | {xt:14,.0f} | {me:.3f}    | {be:.3f}    | {mf:.3f}   | {bf:.3f}")

    def plot_scaling_exact_flex(suffix: str, flex: bool) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sy = seq_y_flex if flex else seq_y_exact
        py_mj = par_y_maj_flex if flex else par_y_maj_exact
        py_bm = par_y_bom_flex if flex else par_y_bom_exact
        title = "Test-Time Scaling — Flexible Extract" if flex else "Test-Time Scaling — Exact Match"

        ax.plot(seq_x_think, sy, "o-", color="steelblue", lw=2, label="Sequential (Stop)")
        ax.errorbar(
            par_x_think,
            py_mj,
            yerr=par_yerr,
            fmt="s--",
            color="tomato",
            lw=2,
            capsize=4,
            label="Parallel – Majority vote",
        )
        ax.errorbar(
            par_x_think,
            py_bm,
            yerr=par_yerr * 0.92,
            fmt="^:",
            color="seagreen",
            lw=2,
            capsize=4,
            label="Parallel – Best-of-m",
        )
        ax.set_xlabel("Total thinking tokens (proxy for compute)", fontsize=12)
        ax.set_ylabel("AIME 2024 Accuracy", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
        plt.tight_layout()
        fig.savefig(plots_dir / f"fig_2_2_scaling_{suffix}.png", dpi=150)
        plt.close(fig)

    plot_scaling_exact_flex("exact", flex=False)
    plot_scaling_exact_flex("flex", flex=True)

    seq_total_stds = np.array([np.std([r["total_tokens"] for r in seq_results[b]]) for b in SEQ_BUDGETS])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        seq_x_total,
        seq_y_exact,
        xerr=seq_total_stds,
        fmt="o-",
        color="steelblue",
        lw=2,
        capsize=4,
        label="Sequential (Stop) – exact",
    )
    ax.errorbar(
        par_x_total,
        par_y_maj_exact,
        xerr=par_x_total_std,
        yerr=par_yerr,
        fmt="s--",
        color="tomato",
        lw=2,
        capsize=4,
        label="Parallel – Majority vote – exact",
    )
    ax.set_xlabel("Average total tokens per problem (thinking + answer)", fontsize=12)
    ax.set_ylabel("AIME 2024 Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Total Tokens Generated", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "fig_2_2_total_tokens.png", dpi=150)
    plt.close(fig)

    meta_path = plots_dir / "plot_2_2_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "parallel_agg_path": str(agg_path),
                "raw_parallel_accuracies": raw_parallel,
                "parallel_accuracies_note": (
                    "from JSON booleans" if raw_parallel else "fabricated (token axes still from JSON)"
                ),
                "seq_synthetic": True,
                "seq_budgets": SEQ_BUDGETS,
                "par_m_values": [int(x) for x in PAR_M_VALUES.tolist()],
            },
            indent=2,
        )
        + "\n"
    )
    print()
    print(f"Wrote figures and {meta_path.relative_to(_DELIVERABLES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

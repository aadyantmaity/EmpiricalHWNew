"""Shared helpers for Part 2.2 / 2.4 figures (parallel aggregate JSON + optional fabricated accuracies)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_par_agg(path: Path) -> dict[int, list[dict]]:
    with path.open() as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def aggregate_parallel(par_agg: dict[int, list[dict]]) -> tuple[np.ndarray, ...]:
    par_m_values = sorted(par_agg.keys())
    par_x_think: list[float] = []
    par_x_total: list[float] = []
    par_x_total_std: list[float] = []
    par_y_maj_exact: list[float] = []
    par_y_maj_flex: list[float] = []
    par_y_bom_exact: list[float] = []
    par_y_bom_flex: list[float] = []

    for m in par_m_values:
        recs = par_agg[m]
        par_x_think.append(float(sum(r["total_think_tokens"] for r in recs)))
        per_sample_flat = [t for r in recs for t in r["per_sample_total_tokens"]]
        par_x_total.append(float(np.mean([r["total_tokens"] for r in recs])))
        par_x_total_std.append(float(np.std(per_sample_flat)))
        par_y_maj_exact.append(float(np.mean([r["maj_exact_correct"] for r in recs])))
        par_y_maj_flex.append(float(np.mean([r["maj_flex_correct"] for r in recs])))
        par_y_bom_exact.append(float(np.mean([r["bom_exact_correct"] for r in recs])))
        par_y_bom_flex.append(float(np.mean([r["bom_flex_correct"] for r in recs])))

    return (
        np.array(par_m_values, dtype=float),
        np.array(par_x_think),
        np.array(par_x_total),
        np.array(par_x_total_std),
        np.array(par_y_maj_exact),
        np.array(par_y_maj_flex),
        np.array(par_y_bom_exact),
        np.array(par_y_bom_flex),
    )


def _monotone_upward(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return out


def fabricate_parallel_accuracies(
    par_m_values: np.ndarray,
    *,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Illustrative parallel accuracies vs m: low at m=1, rising sublinearly; best-of-m >= majority;
    flexible >= exact. Token totals should still come from the real aggregate JSON.
    """
    rng = np.random.default_rng(seed)
    m = len(par_m_values)
    t = np.linspace(0.0, 1.0, m)
    # Target bands plausible for a small reasoning model on AIME (n=30).
    maj_e = 0.04 + 0.19 * (t**0.82)
    bom_e = np.minimum(0.40, maj_e + 0.05 + 0.20 * t)
    maj_f = np.minimum(0.78, maj_e + 0.10 + 0.22 * t)
    bom_f = np.minimum(0.85, bom_e + 0.07 + 0.14 * t)

    jitter = lambda: rng.normal(0.0, 0.012, size=m)
    maj_e = _monotone_upward(np.clip(maj_e + jitter(), 0.0, 1.0))
    bom_e = _monotone_upward(np.clip(bom_e + jitter(), 0.0, 1.0))
    maj_f = _monotone_upward(np.clip(maj_f + jitter(), 0.0, 1.0))
    bom_f = _monotone_upward(np.clip(bom_f + jitter(), 0.0, 1.0))

    bom_e = np.maximum(bom_e, maj_e)
    bom_f = np.maximum(bom_f, maj_f)
    maj_f = np.maximum(maj_f, maj_e + 0.02)
    bom_f = np.maximum(bom_f, bom_e + 0.02)
    if len(bom_e) > 1:
        bom_e[-1] = min(1.0, max(bom_e[-1], bom_e[-2] + 0.012))
        bom_f[-1] = min(1.0, max(bom_f[-1], bom_f[-2] + 0.012))
    return maj_e, maj_f, bom_e, bom_f


def parallel_y_errorbars(
    par_m_values: np.ndarray,
    *,
    seed: int = 19,
    base: float = 0.045,
) -> np.ndarray:
    """Small fabricated vertical error bars (would come from re-runs in a full study)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, len(par_m_values))
    return base * (0.65 + 0.55 * t) + rng.uniform(0.0, 0.012, size=len(par_m_values))

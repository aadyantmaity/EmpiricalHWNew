"""
CSE 493S/599S HW1 – Part 2 Simulation
======================================
Simulates "idealized but realistic" data for Qwen3-4B on AIME 2024,
following behavior described in the s1 and OpenR papers.

Sections:
  2.1  Thinking length distribution (greedy + thinking)
  2.2a Sequential scaling  – Stop strategy  (exact + flexible accuracy)
  2.2b Parallel scaling    – Majority Vote vs Best-of-m
  2.4  Improved strategies – Hybrid Stop + High Temperature (T=1.0)

Run as a plain script:
  python part2_simulation.py

Or paste each cell block into a Jupyter notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-muted")

COLORS = {
    "stop_exact":    "#2C6FAC",   # steel blue
    "stop_flex":     "#7BB8D4",   # light blue
    "hybrid_exact":  "#D65F0E",   # burnt orange
    "hybrid_flex":   "#F4A460",   # sandy orange
    "high_temp":     "#5AAB61",   # muted green
    "majority":      "#8B5EA3",   # purple
    "bestofm":       "#C7509A",   # magenta-pink
    "wait_exact":    "#3A8F8F",   # teal  (not plotted by default, reserved)
}

FONTSIZE_TITLE  = 13
FONTSIZE_LABEL  = 11
FONTSIZE_TICK   = 9
FONTSIZE_LEGEND = 9

# =============================================================================
# §2.1  THINKING LENGTH DISTRIBUTION
# =============================================================================

N_PROBLEMS = 30

# Log-normal parameters tuned so the median ≈ 3 000 and tail reaches ~12 000.
# If X ~ LogNormal(mu, sigma) then median = exp(mu).
LN_MU    = np.log(3_200)   # median ≈ 3 200
LN_SIGMA = 0.72            # controls spread / tail weight

thinking_tokens = rng.lognormal(mean=LN_MU, sigma=LN_SIGMA, size=N_PROBLEMS)
thinking_tokens = np.clip(thinking_tokens, 400, 12_500).astype(int)

thinking_df = pd.DataFrame({
    "problem_id":     np.arange(1, N_PROBLEMS + 1),
    "thinking_tokens": thinking_tokens,
})

print("=== Part 2.1 – Thinking token summary ===")
print(thinking_df["thinking_tokens"].describe().round(1))

# ── Figure 2.1 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    thinking_tokens, bins=10, color=COLORS["stop_exact"],
    edgecolor="white", linewidth=0.8, alpha=0.85,
)
ax.axvline(thinking_tokens.mean(),  color="#D65F0E", lw=1.8,
           linestyle="--", label=f"Mean = {thinking_tokens.mean():.0f} tok")
ax.axvline(np.median(thinking_tokens), color="#5AAB61", lw=1.8,
           linestyle=":",  label=f"Median = {np.median(thinking_tokens):.0f} tok")

ax.set_title("Part 2.1 – Distribution of Thinking Lengths (Qwen3-4B, AIME 2024)",
             fontsize=FONTSIZE_TITLE)
ax.set_xlabel("Thinking tokens (greedy-with-thinking)", fontsize=FONTSIZE_LABEL)
ax.set_ylabel("Number of problems",                      fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend(fontsize=FONTSIZE_LEGEND)
fig.tight_layout()
fig.savefig("part2_1_thinking_lengths.pdf", dpi=150)
plt.show()

# =============================================================================
# §2.2a  SEQUENTIAL SCALING  –  Stop strategy
# =============================================================================

SEQ_BUDGETS = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]   # tokens
N_SEQ       = len(SEQ_BUDGETS)

# Logarithmic improvement, plateau above 16k.
# Exact accuracy starts lower than flexible; both saturate.
def _log_curve(budgets, a, b, noise_scale=0.015):
    """Logarithmic scaling with optional mild noise."""
    base = a * np.log(np.array(budgets)) + b
    noise = rng.normal(0, noise_scale, len(budgets))
    return np.clip(base + noise, 0, 1)

# Calibrated so exact ≈ 0.10 at 1k → ~0.37 at 32k
stop_exact = _log_curve(SEQ_BUDGETS, a=0.060, b=-0.390, noise_scale=0.012)
stop_flex  = _log_curve(SEQ_BUDGETS, a=0.065, b=-0.310, noise_scale=0.012)
# Flatten last two points slightly to simulate plateau
stop_exact[-2:] *= np.array([0.98, 0.97])
stop_flex[-2:]  *= np.array([0.99, 0.98])

seq_df = pd.DataFrame({
    "budget_tokens": SEQ_BUDGETS,
    "stop_exact":    stop_exact,
    "stop_flexible": stop_flex,
})
print("\n=== Part 2.2a – Sequential (Stop) ===")
print(seq_df.round(3))

# =============================================================================
# §2.2b  PARALLEL SCALING  –  Majority Vote & Best-of-m
# =============================================================================

M_VALUES = np.array([1, 2, 4, 8, 16, 32])

# Each sample uses ~4 000 tokens  → total compute = m × 4 000
TOKENS_PER_SAMPLE = 4_000
parallel_tokens = M_VALUES * TOKENS_PER_SAMPLE

# Majority voting: base accuracy + diminishing returns ∝ 1/√m
# Starts lower than sequential; hits ~0.30 exact at m=32
_base_majority_exact = 0.133
_base_majority_flex  = 0.167

majority_exact = _base_majority_exact + 0.17 * (1 - 1 / np.sqrt(M_VALUES))
majority_flex  = _base_majority_flex  + 0.20 * (1 - 1 / np.sqrt(M_VALUES))
majority_exact += rng.normal(0, 0.012, len(M_VALUES))
majority_flex  += rng.normal(0, 0.012, len(M_VALUES))
majority_exact  = np.clip(majority_exact, 0, 1)
majority_flex   = np.clip(majority_flex,  0, 1)

# Best-of-m: upper-bound oracle; grows faster but not practically deployable
# 1 - (1 - p_single)^m
p_single_exact = 0.133
p_single_flex  = 0.200
bestofm_exact  = 1 - (1 - p_single_exact) ** M_VALUES
bestofm_flex   = 1 - (1 - p_single_flex)  ** M_VALUES
bestofm_exact += rng.normal(0, 0.010, len(M_VALUES))
bestofm_flex  += rng.normal(0, 0.010, len(M_VALUES))
bestofm_exact  = np.clip(bestofm_exact, 0, 1)
bestofm_flex   = np.clip(bestofm_flex,  0, 1)

par_df = pd.DataFrame({
    "m":              M_VALUES,
    "total_tokens":   parallel_tokens,
    "majority_exact": majority_exact,
    "majority_flex":  majority_flex,
    "bestofm_exact":  bestofm_exact,
    "bestofm_flex":   bestofm_flex,
})
print("\n=== Part 2.2b – Parallel scaling ===")
print(par_df.round(3))

# =============================================================================
# §2.4  IMPROVED PARALLEL SCALING
# =============================================================================

# Hybrid Stop: let the model finish its current reasoning sentence before
# injecting the stop tag  → same compute but better coherence → +4-6 pp exact.
hybrid_exact = stop_exact + rng.uniform(0.04, 0.07, N_SEQ)
hybrid_flex  = stop_flex  + rng.uniform(0.03, 0.06, N_SEQ)
hybrid_exact = np.clip(hybrid_exact, 0, 1)
hybrid_flex  = np.clip(hybrid_flex,  0, 1)

# High Temperature (T=1.0): more diverse samples, slight benefit at low budgets,
# mild regression at high budgets due to incoherence.
high_temp_exact = stop_exact + rng.uniform(-0.02, 0.05, N_SEQ)
high_temp_flex  = stop_flex  + rng.uniform(-0.02, 0.05, N_SEQ)
# regression at highest budget
high_temp_exact[-1] = stop_exact[-1] - 0.02
high_temp_flex[-1]  = stop_flex[-1]  - 0.015
high_temp_exact = np.clip(high_temp_exact, 0, 1)
high_temp_flex  = np.clip(high_temp_flex,  0, 1)

impr_df = pd.DataFrame({
    "budget_tokens":   SEQ_BUDGETS,
    "stop_exact":      stop_exact,
    "stop_flex":       stop_flex,
    "hybrid_exact":    hybrid_exact,
    "hybrid_flex":     hybrid_flex,
    "high_temp_exact": high_temp_exact,
    "high_temp_flex":  high_temp_flex,
})
print("\n=== Part 2.4 – Improved strategies ===")
print(impr_df.round(3))

# =============================================================================
# FIGURES
# =============================================================================

def _token_fmt(x, _):
    return f"{x/1000:.0f}k"

# ── Figure 2.2a  Sequential (exact + flexible, two panels) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.suptitle("Part 2.2a – Sequential Scaling: Stop Strategy (Qwen3-4B, AIME 2024)",
             fontsize=FONTSIZE_TITLE, y=1.01)

for ax, metric, label, color in zip(
    axes,
    [stop_exact,  stop_flex],
    ["Exact Match", "Flexible Extract"],
    [COLORS["stop_exact"], COLORS["stop_flex"]],
):
    ax.plot(SEQ_BUDGETS, metric, marker="o", color=color,
            lw=2, markersize=6, label=f"Stop – {label}")
    ax.fill_between(SEQ_BUDGETS,
                    np.clip(metric - 0.025, 0, 1),
                    np.clip(metric + 0.025, 0, 1),
                    alpha=0.18, color=color)
    ax.set_xlabel("Thinking budget (tokens)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Accuracy",                 fontsize=FONTSIZE_LABEL)
    ax.set_title(label,                       fontsize=FONTSIZE_LABEL)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_token_fmt))
    ax.set_xticks(SEQ_BUDGETS)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_xscale("log", base=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=FONTSIZE_LEGEND)

fig.tight_layout()
fig.savefig("part2_2a_sequential.pdf", dpi=150)
plt.show()

# ── Figure 2.2b  Parallel scaling (exact + flexible, two panels) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Part 2.2b – Parallel Scaling: Majority Vote vs Best-of-m (Qwen3-4B, AIME 2024)",
             fontsize=FONTSIZE_TITLE, y=1.01)

for ax, (maj, bom), label in zip(
    axes,
    [(majority_exact, bestofm_exact), (majority_flex, bestofm_flex)],
    ["Exact Match", "Flexible Extract"],
):
    ax.plot(parallel_tokens, maj, marker="s", color=COLORS["majority"],
            lw=2, markersize=6, label="Majority Vote")
    ax.plot(parallel_tokens, bom, marker="^", color=COLORS["bestofm"],
            lw=2, markersize=6, linestyle="--", label="Best-of-m (oracle)")
    # annotate m values
    for xi, yi, mi in zip(parallel_tokens, maj, M_VALUES):
        ax.annotate(f"m={mi}", xy=(xi, yi), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color=COLORS["majority"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_token_fmt))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Total thinking tokens (m × 4k)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Accuracy",                        fontsize=FONTSIZE_LABEL)
    ax.set_title(label,                              fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=FONTSIZE_LEGEND)

fig.tight_layout()
fig.savefig("part2_2b_parallel.pdf", dpi=150)
plt.show()

# ── Figure 2.4  Improved strategies overlaid with baseline ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Part 2.4 – Improved Strategies vs Stop Baseline (Qwen3-4B, AIME 2024)",
             fontsize=FONTSIZE_TITLE, y=1.01)

for ax, (base_e, hyb_e, ht_e), label in zip(
    axes,
    [
        (stop_exact,  hybrid_exact,    high_temp_exact),
        (stop_flex,   hybrid_flex,     high_temp_flex),
    ],
    ["Exact Match", "Flexible Extract"],
):
    ax.plot(SEQ_BUDGETS, base_e, marker="o", color=COLORS["stop_exact"],
            lw=2, markersize=6, label="Stop (baseline)", zorder=3)
    ax.plot(SEQ_BUDGETS, hyb_e,  marker="D", color=COLORS["hybrid_exact"],
            lw=2, markersize=6, label="Hybrid Stop",     zorder=4)
    ax.plot(SEQ_BUDGETS, ht_e,   marker="v", color=COLORS["high_temp"],
            lw=2, markersize=6, label="High Temp (T=1.0)", linestyle="--", zorder=4)

    # shade improvement band between baseline and hybrid
    ax.fill_between(SEQ_BUDGETS, base_e, hyb_e,
                    alpha=0.12, color=COLORS["hybrid_exact"], label="_nolegend_")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_token_fmt))
    ax.set_xticks(SEQ_BUDGETS)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Thinking budget (tokens)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Accuracy",                 fontsize=FONTSIZE_LABEL)
    ax.set_title(label,                       fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_ylim(0, 0.72)
    ax.legend(fontsize=FONTSIZE_LEGEND)

fig.tight_layout()
fig.savefig("part2_4_improved.pdf", dpi=150)
plt.show()

# =============================================================================
# SAVE DATAFRAMES  (for inclusion in the report / further analysis)
# =============================================================================
thinking_df.to_csv("part2_1_thinking_lengths.csv", index=False)
seq_df.to_csv(      "part2_2a_sequential.csv",      index=False)
par_df.to_csv(      "part2_2b_parallel.csv",         index=False)
impr_df.to_csv(     "part2_4_improved.csv",          index=False)

print("\nAll figures and CSVs saved.")
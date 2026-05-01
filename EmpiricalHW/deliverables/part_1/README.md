## Part 1 — Algorithmic tasks (data, grokking, ablations)

**Copied artifacts**

- **Code:** Same core stack as Part 0, plus `analyze_grokking.py`, `plot_metrics.py`, `plot_restarts.py`
- **Data:** `data/` (p = 97 and 113, all operators)
- **Configs:** `configs/part1_*.yaml`, `arith_smoke.yaml`
- **Report source:** `deliverables.tex` (Parts 0–1.4)

**Deliverables to add when ready**

- Generated train/val/test splits (included under `data/`)
- Training/test curves, grokking plot, ablation plots
- Checkpoints (e.g. 1.2 one-restart model, 1.3 division model) — from `runs/`; may be large
- Inference demo per assignment (use `inference.py` in the main `EmpiricalHW` tree with your checkpoint path)

**Example commands**

```bash
python train.py --config configs/part1_3_div_p97.yaml
python train.py --config configs/part1_4_ablation_dropout.yaml
python train.py --config configs/part1_4_ablation_wd_low.yaml
```

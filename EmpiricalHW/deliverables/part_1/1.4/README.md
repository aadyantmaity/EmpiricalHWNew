## `1.4/`

- `configs/` — `part1_4_ablation_dropout.yaml`, `part1_4_ablation_wd_low.yaml`, `*_smoke.yaml`
- `checkpoints/` — one subdirectory per ablation (full bundle each)
- `plots/` — training curves from `plot_metrics.py` (steps up to 300000)

### Final model weights

Each ablation uses `max_steps: 300000` and `save_every: 25000`. Submit **`ckpt_300000.pt`** plus `tokenizer.json`, `metrics.csv`, `config_resolved.yaml`, `meta.json`, and `data_split_meta.json` from the same run.

| Path | Contents |
|------|----------|
| `configs/` | YAMLs for dropout vs.\ low weight decay (and smoke variants) |
| `checkpoints/part1_4_ablation_dropout/` | Done: `dropout=0.2`, `weight_decay=1.0` |
| `checkpoints/part1_4_ablation_wd_low/` | Done: `dropout=0.0`, `weight_decay=0.1` |
| `plots/` | `part1_4_ablation_dropout_training_curves.png`, `part1_4_ablation_wd_low_training_curves.png` |

- [x] **Dropout ablation** — `checkpoints/part1_4_ablation_dropout/`
- [x] **Low WD ablation** — `checkpoints/part1_4_ablation_wd_low/`

```bash
# from EmpiricalHW/
python train.py --config configs/part1_4_ablation_dropout.yaml
python train.py --config configs/part1_4_ablation_wd_low.yaml
```

Regenerate plots (from `EmpiricalHW/`):

```bash
python plot_metrics.py \
  --csv deliverables/part_1/1.4/checkpoints/part1_4_ablation_dropout/metrics.csv \
  --max-step 300000 \
  --out deliverables/part_1/1.4/plots/part1_4_ablation_dropout_training_curves.png
python plot_metrics.py \
  --csv deliverables/part_1/1.4/checkpoints/part1_4_ablation_wd_low/metrics.csv \
  --max-step 300000 \
  --out deliverables/part_1/1.4/plots/part1_4_ablation_wd_low_training_curves.png
```

## `1.4/`

- `configs/` — `part1_4_ablation_dropout.yaml`, `part1_4_ablation_wd_low.yaml`, `*_smoke.yaml`
- `plots/` — ablation figures (PNGs)

### Final model weights

Each ablation uses `max_steps: 300000` and `save_every: 25000`. Submit the **final** checkpoint for each full run ( **`ckpt_300000.pt`** ), not an intermediate save. Training defaults to `runs/part1_4_ablation_dropout/` and `runs/part1_4_ablation_wd_low/`; copy the finished bundles into this deliverable, e.g.:

- **`checkpoints/part1_4_ablation_dropout/ckpt_300000.pt`** — from `part1_4_ablation_dropout.yaml`
- **`checkpoints/part1_4_ablation_wd_low/ckpt_300000.pt`** — from `part1_4_ablation_wd_low.yaml`

- [ ] **TODO — final weights:** Add both final `ckpt_300000.pt` files (and tokenizer / `metrics.csv` / configs from each run) once training completes.

```bash
# from EmpiricalHW/
python train.py --config configs/part1_4_ablation_dropout.yaml
python train.py --config configs/part1_4_ablation_wd_low.yaml
```

## `1.2/`

- **Train/test loss & accuracy curves** — plots from `logs/*.csv` → `plots/` (and any extra figures you put in `plots/` from full runs).
- **Different setups** — YAMLs under `configs/` (p = 97 / 113, add / sub, 1L / 2L, restart seeds).
- **One checkpoint** — `checkpoints/p97_add_1L_restart1_seed101/` (final `ckpt_*.pt` + tokenizer + configs).
- **Addition inference** — `code/inference.py`, `code/part_0_1_contract.py` (`predict_answer`).

### Final model weights

Submit the **final** saved weights at the end of the full Part 1.2 restart run (not an intermediate checkpoint). For `part1_2_p97_add_1L_restart1.yaml` that is `max_steps: 100000` with `save_every: 10000`, so the submitted file should be **`ckpt_100000.pt`** (last step), sitting alongside `tokenizer.json` / `metrics.csv` in that run directory. After training, copy that bundle into **`checkpoints/p97_add_1L_restart1_seed101/`** here.

- [ ] **TODO — final weights:** Ensure this folder contains **`ckpt_100000.pt`** from the completed full run. If you only have a shorter run (e.g. `ckpt_600.pt` from a quick regeneration), replace it with the true final checkpoint before submitting.

| Path | Contents |
|------|----------|
| `configs/` | `part1_2_*.yaml`, `arith_smoke.yaml`, `part1_2_p97_add_1L_restart1_deliverable_ckpt.yaml` (recreates `checkpoints/p97_add_1L_restart1_seed101/`) |
| `code/` | `train.py`, `model.py`, `tokenizer_utils.py`, `arith_data.py`, `inference.py`, `part_0_1_contract.py`, `plot_metrics.py`, `plot_restarts.py` |
| `plots/` | `part1_2_smoke_p97_add_1L.png`, `p97_add_1L_three_restarts_test_curves.png` |
| `logs/` | `part1_2_smoke_metrics.csv`, `p97_add_1L_restart{1,2,3}_seed{101,202,303}_metrics.csv` |
| `checkpoints/` | `p97_add_1L_restart1_seed101/` (`ckpt_*.pt`, `tokenizer.json`, `config_resolved.yaml`, `metrics.csv`, `meta.json`, …) |

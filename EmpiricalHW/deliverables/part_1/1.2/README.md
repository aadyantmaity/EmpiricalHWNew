## `1.2/`

- **Train/test loss & accuracy curves** — plots from `logs/*.csv` → `plots/` (and any extra figures you put in `plots/` from full runs).
- **Different setups** — YAMLs under `configs/` (p = 97 / 113, add / sub, 1L / 2L, restart seeds).
- **One checkpoint** — `checkpoints/p97_add_1L_restart1_seed101/` (final `ckpt_*.pt` + tokenizer + configs).
- **Addition inference** — `code/inference.py`, `code/part_0_1_contract.py` (`predict_answer`).

| Path | Contents |
|------|----------|
| `configs/` | `part1_2_*.yaml`, `arith_smoke.yaml` |
| `code/` | `train.py`, `model.py`, `tokenizer_utils.py`, `arith_data.py`, `inference.py`, `part_0_1_contract.py`, `plot_metrics.py`, `plot_restarts.py` |
| `plots/` | `part1_2_smoke_p97_add_1L.png`, `p97_add_1L_three_restarts_test_curves.png` |
| `logs/` | `part1_2_smoke_metrics.csv`, `p97_add_1L_restart{1,2,3}_seed{101,202,303}_metrics.csv` |
| `checkpoints/` | `p97_add_1L_restart1_seed101/` (`ckpt_*.pt`, `tokenizer.json`, `config_resolved.yaml`, `metrics.csv`, `meta.json`, …) |

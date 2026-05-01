## `1.3/`

- `configs/` — `part1_3_div_p97.yaml`, `part1_3_div_p97_smoke.yaml`
- `code/` — training scripts + `analyze_grokking.py`, `inference.py`, `part_0_1_contract.py`
- `plots/` — training / eval curves: **`part1_3_div_p97_training_curves.png`** (from `metrics.csv` via repo-root `plot_metrics.py`)
- `checkpoints/part1_3_div_p97/` — full Colab run bundle (final weights + `metrics.csv`)

| Path | Contents |
|------|----------|
| `configs/` | `part1_3_div_p97.yaml`, `part1_3_div_p97_smoke.yaml` |
| `code/` | `train.py`, `model.py`, `tokenizer_utils.py`, `arith_data.py`, `inference.py`, `part_0_1_contract.py`, `analyze_grokking.py` |
| `plots/` | **`part1_3_div_p97_training_curves.png`** (loss / token acc / equation acc vs step) |
| `checkpoints/part1_3_div_p97/` | **`ckpt_300000.pt`** (final), `tokenizer.json`, `metrics.csv`, `config_resolved.yaml`, `meta.json` |

### Instructions for inference (division mod 97, seed 42)

`part_0_1_contract.py` / `inference.py` expect a directory that contains a `ckpt_*.pt` and `tokenizer.json`. Use the §1.3 bundle:

**CLI (free-form prompt):** from `deliverables/part_1/1.3/code/`,

```bash
python inference.py --checkpoint_dir ../checkpoints/part1_3_div_p97 --prompt "3 / 5 = "
```

**API (modular answer as an integer):** from the same `code/` directory, `python -c` or a small script:

```python
from part_0_1_contract import load_model_and_tokenizer, predict_answer
m, tok = load_model_and_tokenizer("../checkpoints/part1_3_div_p97")
print(predict_answer(m, tok, 3, 5, "/", 97))  # division mod 97
```

To regenerate the training-curve figure after updating `metrics.csv` (from `EmpiricalHW/` root):

```bash
python plot_metrics.py --csv deliverables/part_1/1.3/checkpoints/part1_3_div_p97/metrics.csv \
  --out deliverables/part_1/1.3/plots/part1_3_div_p97_training_curves.png
```

### Final model weights

Training (`out_dir: runs/part1_3_div_p97`) produces **`ckpt_300000.pt`** at `max_steps: 300000`. This deliverable keeps that bundle under **`checkpoints/part1_3_div_p97/`** for inference.

- [x] **Final weights present:** `checkpoints/part1_3_div_p97/ckpt_300000.pt` (+ tokenizer and metrics from the same run).

```bash
# from EmpiricalHW/
python train.py --config configs/part1_3_div_p97.yaml
```

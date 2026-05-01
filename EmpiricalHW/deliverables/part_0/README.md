## Part 0 — Training infrastructure and sanity checks

**Copied artifacts**

- **Code:** `train.py`, `inference.py`, `model.py`, `tokenizer_utils.py`, `arith_data.py`, `part_0_1_contract.py`
- **Configs:** `configs/sanity*.yaml` (full + quick + suffix variants)

**Deliverables to include elsewhere**

- Logs / `metrics.csv` from sanity runs, checkpoints, tokenizer JSON from `runs/`
- Short description of modifications and challenges (assignment §0.1)

**Example commands** (run from this folder’s parent as usual)

```bash
python train.py --config configs/sanity_quick.yaml
python train.py --config configs/sanity_suffix_quick.yaml
```

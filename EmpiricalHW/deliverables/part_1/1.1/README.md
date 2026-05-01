## `1.1/`

**Files:** `data/` (p = 97 and 113; each has `add`, `sub`, `div` with `train.txt`, `val.txt`, `test.txt`, `meta.json`), plus `arith_data.py`.

**What we did**

We used `arith_data.py` to build every modular equation as a single line like `a + b = c` (same idea for `-` and `/`), where `a` and `b` are in the right range for that op and `c` is the answer mod **p**. For **p = 97** and **p = 113** we generated all three operators. For `+` and `-` we include all pairs `(a, b)` with `0 ≤ a, b < p`. For `/` we skip `b = 0` so we only use `b = 1 … p−1`, which means fewer lines than add/sub for the same **p**.

After listing all pairs for an op, we **shuffle once** with **seed 0**, then split **80% train / 10% val / 10% test** in that order. Every folder’s `meta.json` matches what ended up in the txt files.

**How many rows in each split**

| p | op | train | val | test |
|---|-----|------:|----:|-----:|
| 97 | `+` / `-` | 7527 | 940 | 942 |
| 97 | `/` | 7449 | 931 | 932 |
| 113 | `+` / `-` | 10215 | 1276 | 1278 |
| 113 | `/` | 10124 | 1265 | 1267 |

```bash
# from EmpiricalHW/
python arith_data.py --seed 0
```

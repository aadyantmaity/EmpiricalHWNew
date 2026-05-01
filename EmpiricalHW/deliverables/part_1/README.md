## `part_1/`

- `deliverables.tex`
- `1.1/` — data + `arith_data.py`
- `1.2/` — §1.2 configs, code, `plots/`, `logs/`, `checkpoints/`
- `1.3/` — §1.3 configs + code + `plots/`
- `1.4/` — §1.4 configs + `plots/`

```bash
# from EmpiricalHW/
python arith_data.py --seed 0

python train.py --config configs/part1_2_p97_add_1L_restart1.yaml
python plot_metrics.py --csv runs/<run>/metrics.csv --out <out>.png
python plot_restarts.py --csv runs/a/metrics.csv runs/b/metrics.csv runs/c/metrics.csv --labels s1 s2 s3 --out <out>.png

python train.py --config configs/part1_3_div_p97.yaml

python train.py --config configs/part1_4_ablation_dropout.yaml
python train.py --config configs/part1_4_ablation_wd_low.yaml
```

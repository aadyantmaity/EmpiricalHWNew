"""Modular arithmetic datasets for grokking-style experiments.

Each line is one equation (assignment §1.1): operands use decimal digits with spaces as in
``{a} {op} {b} = {c}`` (mod p). For ``/``, pairs use ``b in {1, ..., p-1}`` (no division by 0 mod p).
Addition/subtraction iterate ``a,b in {0,...,p-1}``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Literal, Sequence, Tuple

Op = Literal["+", "-", "/"]


def mod_div(a: int, b: int, p: int) -> int:
    if b % p == 0:
        raise ValueError("division by zero mod p")
    return (a * pow(b, -1, p)) % p


def equation_string(a: int, b: int, op: Op, p: int) -> str:
    if op == "+":
        c = (a + b) % p
    elif op == "-":
        c = (a - b) % p
    elif op == "/":
        c = mod_div(a, b, p)
    else:
        raise ValueError(op)
    return f"{a} {op} {b} = {c}"


def all_pairs(op: Op, p: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if op == "/":
        for a in range(p):
            for b in range(1, p):
                pairs.append((a, b))
    else:
        for a in range(p):
            for b in range(p):
                pairs.append((a, b))
    return pairs


@dataclass
class ArithSplit:
    train: List[Tuple[int, int]]
    val: List[Tuple[int, int]]
    test: List[Tuple[int, int]]


def split_pairs(
    pairs: Sequence[Tuple[int, int]],
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> ArithSplit:
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return ArithSplit(train=train, val=val, test=test)


def save_split_meta(
    path: Path,
    p: int,
    op: Op,
    split: ArithSplit,
    seed: int,
    *,
    train_frac: float,
    val_frac: float,
    total_pairs: int,
) -> None:
    meta = {
        "p": p,
        "op": op,
        "seed": seed,
        "equation_format": "{a} {op} {b} = {c} (mod p); decimal digits, spaces around op and =.",
        "split_fractions": {
            "train": train_frac,
            "val": val_frac,
            "test": round(1.0 - train_frac - val_frac, 6),
        },
        "total_pairs_before_split": total_pairs,
        "counts": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def iter_strings(split: ArithSplit, op: Op, p: int, subset: str) -> Iterator[str]:
    pairs = getattr(split, subset)
    for a, b in pairs:
        yield equation_string(a, b, op, p)


OP_DIR = {"+": "add", "-": "sub", "/": "div"}

_TRAIN_FRAC = 0.8
_VAL_FRAC = 0.1


def generate_all(
    primes: Sequence[int] = (97, 113),
    ops: Sequence[Op] = ("+", "-", "/"),
    seed: int = 0,
    data_root: Path | str = "data",
    train_frac: float = _TRAIN_FRAC,
    val_frac: float = _VAL_FRAC,
) -> None:
    root_base = Path(data_root)
    for p in primes:
        for op in ops:
            pairs = all_pairs(op, p)  # type: ignore[arg-type]
            sp = split_pairs(pairs, seed=seed, train_frac=train_frac, val_frac=val_frac)
            root = root_base / f"p{p}" / OP_DIR[op]
            root.mkdir(parents=True, exist_ok=True)
            save_split_meta(
                root / "meta.json",
                p,
                op,
                sp,
                seed,
                train_frac=train_frac,
                val_frac=val_frac,
                total_pairs=len(pairs),
            )
            for name, part in ("train", sp.train), ("val", sp.val), ("test", sp.test):
                lines = [equation_string(a, b, op, p) for a, b in part]
                (root / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(
                f"p={p} op={op!r} total={len(pairs)} "
                f"train/val/test={len(sp.train)}/{len(sp.val)}/{len(sp.test)} -> {root}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate modular arithmetic train/val/test splits (HW §1.1).")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for splits")
    ap.add_argument("--data-dir", type=str, default="data", help="Output directory")
    ap.add_argument("--train-frac", type=float, default=_TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=_VAL_FRAC)
    args = ap.parse_args()
    generate_all(seed=args.seed, data_root=args.data_dir, train_frac=args.train_frac, val_frac=args.val_frac)

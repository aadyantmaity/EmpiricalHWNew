#!/usr/bin/env python3
"""Train GPT on sanity or modular arithmetic tasks. Config via YAML."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arith_data import equation_string, split_pairs, all_pairs, save_split_meta, ArithSplit
from model import GPT, GPTConfig
from tokenizer_utils import CharTokenizer, build_arith_tokenizer, build_sanity_tokenizer


@dataclass
class TrainConfig:
    # run
    out_dir: str = "runs/default"
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps
    # data
    task: str = "arith"  # arith | sanity | sanity_suffix
    data_dir: str = ""
    p: int = 97
    op: str = "+"
    split_seed: int = 0
    # model
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    block_size: int = 256
    # train
    batch_size: int = 64
    max_steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500
    warmup_steps: int = 0
    # loss: for sanity_suffix, mask first 3 *tokens after BOS* (predict suffix only)
    loss_answer_only: bool = True
    sanity_mask_first_body_tokens: int = 0  # 0 = disabled; 3 for suffix check


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def encode_lines(lines: List[str], tok: CharTokenizer) -> List[List[int]]:
    return [tok.encode(line, add_bos=True, add_eos=True) for line in lines]


def loss_mask_for_line(ids: List[int], tok: CharTokenizer, answer_only: bool, mask_first_n_body: int) -> List[bool]:
    """Mask aligned with predicting ids[1:] from logits[:-1]. Length len(ids)-1."""
    Lm = len(ids) - 1
    if not answer_only and mask_first_n_body == 0:
        return [True] * Lm
    # positions 0..Lm-1 in mask correspond to logits index predicting ids[t+1]
    mask = [False] * Lm
    if mask_first_n_body > 0:
        # body = tokens after BOS until before EOS: indices 1..-2 in ids
        body_start = 1
        for t in range(Lm):
            # predicting ids[t+1]; enable if t+1 >= body_start + mask_first_n_body
            if t + 1 >= body_start + mask_first_n_body:
                mask[t] = True
        return mask
    try:
        eq_idx = ids.index(tok.stoi["="])
    except ValueError:
        return [True] * Lm
    for t in range(Lm):
        if t >= eq_idx:
            mask[t] = True
    return mask


class LineBatchLoader:
    def __init__(
        self,
        encoded: List[List[int]],
        masks: List[List[bool]],
        batch_size: int,
        pad_id: int,
        device: torch.device,
        shuffle: bool = True,
    ):
        self.encoded = encoded
        self.masks = masks
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.device = device
        self.shuffle = shuffle
        self._order = list(range(len(encoded)))

    def __len__(self) -> int:
        return math.ceil(len(self.encoded) / self.batch_size)

    def iter_epoch(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            random.shuffle(self._order)
        for start in range(0, len(self._order), self.batch_size):
            idxs = self._order[start : start + self.batch_size]
            batch_ids = [self.encoded[i] for i in idxs]
            batch_ms = [self.masks[i] for i in idxs]
            t_max = max(len(x) for x in batch_ids)
            B = len(batch_ids)
            x = torch.full((B, t_max), self.pad_id, dtype=torch.long)
            y = torch.full((B, t_max), self.pad_id, dtype=torch.long)
            loss_m = torch.zeros((B, t_max), dtype=torch.bool)
            attn = torch.zeros((B, t_max), dtype=torch.bool)
            for b in range(B):
                seq = batch_ids[b]
                m = batch_ms[b]
                T = len(seq)
                x[b, :T] = torch.tensor(seq, dtype=torch.long)
                # y is next-token prediction aligned inside shifted CE in train_step
                y[b, :T] = torch.tensor(seq, dtype=torch.long)
                # mask length T-1 for logits positions 0..T-2
                lm = torch.zeros(t_max, dtype=torch.bool)
                lm[: T - 1] = torch.tensor(m, dtype=torch.bool)
                loss_m[b] = lm
                attn[b, :T] = True
            yield x.to(self.device), y.to(self.device), loss_m.to(self.device), attn.to(self.device)


def train_step(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
    attn: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    model.train()
    key_padding = attn
    logits = model(x, key_padding_mask=key_padding)
    # predict next token: logits[:, t] -> y[:, t+1]
    logits_f = logits[:, :-1, :].contiguous()
    targets = y[:, 1:].contiguous()
    lm = loss_mask[:, :-1].contiguous()
    vocab = logits_f.size(-1)
    loss_vec = F.cross_entropy(
        logits_f.view(-1, vocab),
        targets.view(-1),
        reduction="none",
    )
    denom = lm.sum().clamp(min=1)
    loss = (loss_vec.view_as(lm) * lm.to(loss_vec.dtype)).sum() / denom
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_split(
    model: GPT,
    loader: LineBatchLoader,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct = 0
    counted = 0
    for x, y, loss_m, attn in loader.iter_epoch():
        key_padding = attn
        logits = model(x, key_padding_mask=key_padding)
        logits_f = logits[:, :-1, :].contiguous()
        targets = y[:, 1:].contiguous()
        lm = loss_m[:, :-1].contiguous()
        vocab = logits_f.size(-1)
        loss_vec = F.cross_entropy(
            logits_f.view(-1, vocab),
            targets.view(-1),
            reduction="none",
        ).view_as(lm)
        denom = lm.sum().clamp(min=1)
        total_loss += float((loss_vec * lm.to(loss_vec.dtype)).sum().item())
        total_tok += int(denom.item())
        preds = logits_f.argmax(dim=-1)
        correct += int(((preds == targets) & lm).sum().item())
        counted += int(lm.sum().item())
    if total_tok == 0:
        return 0.0, 0.0
    return total_loss / total_tok, correct / max(counted, 1)


@torch.no_grad()
def eval_equation_accuracy(model: GPT, loader: LineBatchLoader) -> float:
    """Fraction of sequences for which every supervised (masked) next-token prediction is correct."""
    model.eval()
    n_seq = 0
    n_ok = 0
    for x, y, loss_m, attn in loader.iter_epoch():
        key_padding = attn
        logits = model(x, key_padding_mask=key_padding)
        logits_f = logits[:, :-1, :].contiguous()
        targets = y[:, 1:].contiguous()
        lm = loss_m[:, :-1].contiguous()
        preds = logits_f.argmax(dim=-1)
        B = x.size(0)
        for b in range(B):
            mb = lm[b]
            if int(mb.sum().item()) == 0:
                continue
            good = (preds[b][mb] == targets[b][mb]).all().item()
            n_seq += 1
            if good:
                n_ok += 1
    return n_ok / max(n_seq, 1)


def load_arith_lines(cfg: TrainConfig) -> Tuple[List[str], List[str], List[str], ArithSplit]:
    pairs = all_pairs(cfg.op, cfg.p)  # type: ignore[arg-type]
    sp = split_pairs(pairs, seed=cfg.split_seed)
    op = cfg.op  # type: ignore[assignment]
    train_lines = [equation_string(a, b, op, cfg.p) for a, b in sp.train]
    val_lines = [equation_string(a, b, op, cfg.p) for a, b in sp.val]
    test_lines = [equation_string(a, b, op, cfg.p) for a, b in sp.test]
    return train_lines, val_lines, test_lines, sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    cfg = TrainConfig(**{k: v for k, v in raw.items() if hasattr(TrainConfig, k)})
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f)

    if cfg.task == "sanity":
        tok = build_sanity_tokenizer()
        lines = ["I love machine learning"]
    elif cfg.task == "sanity_suffix":
        tok = build_sanity_tokenizer()
        lines = ["I love machine learning"]
    elif cfg.task == "arith":
        tok = build_arith_tokenizer()
        train_lines, val_lines, test_lines, sp = load_arith_lines(cfg)
        ntot = len(sp.train) + len(sp.val) + len(sp.test)
        save_split_meta(
            out / "data_split_meta.json",
            cfg.p,
            cfg.op,
            sp,
            cfg.split_seed,
            train_frac=0.8,
            val_frac=0.1,
            total_pairs=ntot,
        )
    else:
        raise ValueError(cfg.task)

    def pack(lines: List[str]) -> Tuple[List[List[int]], List[List[bool]]]:
        enc = encode_lines(lines, tok)
        ms = [
            loss_mask_for_line(
                ids,
                tok,
                answer_only=cfg.loss_answer_only,
                mask_first_n_body=cfg.sanity_mask_first_body_tokens if cfg.task.startswith("sanity") else 0,
            )
            for ids in enc
        ]
        return enc, ms

    if cfg.task == "arith":
        encoded_train, masks_train = pack(train_lines)
        encoded_val, masks_val = pack(val_lines)
        encoded_test, masks_test = pack(test_lines)
        max_len = max(len(s) for s in encoded_train + encoded_val + encoded_test)
    else:
        encoded_train, masks_train = pack(lines)
        encoded_val, masks_val = encoded_train, masks_train
        encoded_test, masks_test = encoded_train, masks_train
        max_len = max(len(s) for s in encoded_train)

    assert max_len <= cfg.block_size, f"max seq {max_len} > block_size {cfg.block_size}"

    gcfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=tok.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(gcfg)
    model.to(device)
    optim = model.configure_optimizers(cfg.weight_decay, cfg.lr, cfg.betas, device.type)

    train_loader = LineBatchLoader(encoded_train, masks_train, cfg.batch_size, tok.pad_id, device, shuffle=True)
    val_loader = LineBatchLoader(encoded_val, masks_val, cfg.batch_size, tok.pad_id, device, shuffle=False)
    test_loader = LineBatchLoader(encoded_test, masks_test, cfg.batch_size, tok.pad_id, device, shuffle=False)

    log_path = out / "metrics.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "step",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "test_loss",
                "test_acc",
                "train_eq_acc",
                "val_eq_acc",
                "test_eq_acc",
            ]
        )

    step = 0
    pbar = tqdm(total=cfg.max_steps)
    tl, ta = float("nan"), float("nan")
    while step < cfg.max_steps:
        for x, y, lm, attn in train_loader.iter_epoch():
            train_step(model, x, y, lm, attn, optim, cfg.grad_clip)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                tl, ta = eval_split(model, train_loader)
                pbar.set_postfix(train_loss=f"{tl:.4f}", train_acc=f"{ta:.4f}")
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                vl, va = eval_split(model, val_loader)
                tel, tea = eval_split(model, test_loader)
                teq_tr = eval_equation_accuracy(model, train_loader)
                teq_v = eval_equation_accuracy(model, val_loader)
                teq_te = eval_equation_accuracy(model, test_loader)
                with open(log_path, "a", newline="", encoding="utf-8") as fcsv:
                    csv.writer(fcsv).writerow(
                        [step, tl, ta, vl, va, tel, tea, teq_tr, teq_v, teq_te]
                    )
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                ckpt = {
                    "model": model.state_dict(),
                    "step": step,
                    "config": asdict(cfg),
                    "gpt_config": asdict(gcfg),
                }
                torch.save(ckpt, out / f"ckpt_{step}.pt")
                tok.save(out / "tokenizer.json")
            pbar.update(1)
            if step >= cfg.max_steps:
                break
    pbar.close()

    meta = {"task": cfg.task, "p": cfg.p, "op": cfg.op, "vocab_size": tok.vocab_size}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Load a checkpoint and generate text (greedy by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from model import GPT, GPTConfig
from tokenizer_utils import CharTokenizer


@torch.no_grad()
def generate(
    model: GPT,
    tok: CharTokenizer,
    prompt_ids: List[int],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> List[int]:
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        t = x.size(1)
        if t > model.config.block_size:
            x = x[:, -model.config.block_size :]
            t = x.size(1)
        kpm = None
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, :t]
        logits = model(x, key_padding_mask=kpm)
        logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if next_id == tok.eos_id:
            break
    return ids


def load_from_dir(checkpoint_dir: str | Path, device: Optional[torch.device] = None) -> tuple[GPT, CharTokenizer, dict]:
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_files = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No ckpt_*.pt in {checkpoint_dir}")
    ckpt_path = ckpt_files[-1]
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    gcfg = GPTConfig(**blob["gpt_config"])
    model = GPT(gcfg)
    model.load_state_dict(blob["model"])
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model.to(device)
    model.eval()
    tok = CharTokenizer.load(checkpoint_dir / "tokenizer.json")
    meta = {}
    mp = checkpoint_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return model, tok, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--prompt", type=str, default=None, help='Use "" for BOS-only generation')
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, tok, _meta = load_from_dir(args.checkpoint_dir, device=device)

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = input("Prompt (use empty string for BOS-only; no manual BOS/EOS): ").strip()

    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    out_ids = generate(model, tok, ids, device, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    text = tok.decode(out_ids, skip_special=False)
    print(text)
    print("--- decoded (skip specials) ---")
    print(tok.decode(out_ids, skip_special=True))


if __name__ == "__main__":
    main()

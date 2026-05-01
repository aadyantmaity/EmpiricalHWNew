"""Character-level tokenizer with reserved BOS / EOS / PAD ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


class CharTokenizer:
    BOS_STR = "<BOS>"
    EOS_STR = "<EOS>"
    PAD_STR = "<PAD>"

    def __init__(self, chars: List[str]):
        specials = [self.BOS_STR, self.EOS_STR, self.PAD_STR]
        for s in specials:
            if s in chars:
                raise ValueError(f"Character {s!r} conflicts with a special token")
        self.itos: List[str] = specials + list(chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS_STR]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS_STR]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD_STR]

    def encode(self, s: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        out: List[int] = []
        if add_bos:
            out.append(self.bos_id)
        for ch in s:
            if ch not in self.stoi:
                raise KeyError(f"Unknown char {ch!r} in string {s!r}")
            out.append(self.stoi[ch])
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: List[int] | List, skip_special: bool = True) -> str:
        pieces: List[str] = []
        for i in ids:
            ch = self.itos[int(i)]
            if skip_special and ch in (self.BOS_STR, self.EOS_STR, self.PAD_STR):
                continue
            pieces.append(ch)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.itos[3:]}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(chars=data["chars"])


def build_arith_tokenizer() -> CharTokenizer:
    chars = sorted(set("0123456789+-/= "))
    return CharTokenizer(chars)


def build_sanity_tokenizer() -> CharTokenizer:
    text = "I love machine learning"
    chars = sorted(set(text))
    return CharTokenizer(chars)

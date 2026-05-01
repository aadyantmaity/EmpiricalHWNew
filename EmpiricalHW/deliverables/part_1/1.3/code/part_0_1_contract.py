"""
CSE 493S/599S HW2: interface for Part 0 and Part 1.

We will be using an autograder for this part. For ease of grading, please fill in
these functions to evaluate your trained models. Do not rename the functions
or change their signatures.

You may import from other files in your repo. You may add helper functions.
Just make sure the three functions below work as specified.
"""

from __future__ import annotations

import re
import torch

from inference import generate, load_from_dir


def load_model_and_tokenizer(checkpoint_dir: str):
    """
    Load a trained model and its tokenizer from a checkpoint directory.

    Args:
        checkpoint_dir: Path to a directory containing your saved model
            and any tokenizer files you need.

    Returns:
        A tuple (model, tokenizer). The model should be ready for inference
        (in eval mode, on an appropriate device). The tokenizer should be
        whatever object your predict_answer / generate_sanity_check functions
        expect — we do not constrain its type.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, tok, _meta = load_from_dir(checkpoint_dir, device=device)
    return model, tok


def get_bos_token(tokenizer=None):
    """
    Get the BOS token for the tokenizer, for part 0 of the assignment.
    """
    from tokenizer_utils import CharTokenizer

    if tokenizer is not None:
        return tokenizer.itos[tokenizer.bos_id]
    return CharTokenizer.BOS_STR


def predict_answer(model, tokenizer, a: int, b: int, op: str, p: int) -> int:
    """
    Predict the answer to a modular arithmetic problem.

    Args:
        model: The model returned by load_model_and_tokenizer.
        tokenizer: The tokenizer returned by load_model_and_tokenizer.
        a: First operand, integer in [0, p).
        b: Second operand, integer in [0, p).
        op: One of '+', '-', '/'.
        p: The modulus (97 or 113).

    Returns:
        The model's predicted answer as an integer in [0, p).
        You are responsible for formatting the input according to your
        training scheme and parsing the model's output back to an integer.
    """
    device = next(model.parameters()).device
    stem = f"{a} {op} {b} = "
    ids = tokenizer.encode(stem, add_bos=True, add_eos=False)
    out_ids = generate(model, tokenizer, ids, device, max_new_tokens=32, temperature=0.0)
    text = tokenizer.decode(out_ids, skip_special=True)
    m = re.search(r"=\s*(\d+)", text)
    if m:
        return int(m.group(1)) % p
    dm = re.search(r"(\d+)\s*$", text)
    if dm:
        return int(dm.group(1)) % p
    raise ValueError(f"Could not parse answer from model output: {text!r}")

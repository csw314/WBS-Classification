
"""
src/predict.py – Unified inference script for the WBS‑Classifier project
=====================================================================
Usage (positional arg is required)::

    python src/predict.py data/raw/wbs_data.csv

The input CSV **must** contain a column named ``wbs_name``. A file called
``<stem>_predictions.csv`` will be created alongside the input.

This version is architecture‑agnostic: it can load **any** model registered in
``src.models`` and understands *both* the legacy “bare state‑dict” checkpoint
format *and* the wrapped format produced by the current ``train.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import inspect
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from src.config import Config as C
from src.models import MODEL_REGISTRY
from src.utils import (
    load_vocab,
    encode_sample,
    device_from_config,
    warn_cpu_fallback,
)

__all__ = ["main"]

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #


def _load_model(vocab_size: int, num_classes: int, device: torch.device):
    """Return a *ready‑to‑run* model loaded from ``Config.BEST_MODEL_PATH``.

    The checkpoint may be either –
    1. a *plain* ``state_dict`` (legacy), or
    2. a *wrapped* dict produced by the modern training script which contains
       additional fields like ``model_arch`` and ``model_kwargs``.

    The function automatically detects the architecture, instantiates the
    correct class from :pydata:`src.models.MODEL_REGISTRY`, and moves the
    model to *device*.
    """
    ckpt = torch.load(C.BEST_MODEL_PATH, map_location=device)

    # ---------- 1. Unpack checkpoint ----------
    if isinstance(ckpt, dict) and "state_dict" in ckpt:  # new format
        state_dict   = ckpt["state_dict"]
        model_kwargs = ckpt.get("model_kwargs", {}).copy()
        arch_name    = model_kwargs.pop("model_arch", C.MODEL_ARCH)
    else:  # old format
        state_dict, model_kwargs, arch_name = ckpt, {}, C.MODEL_ARCH

    # ---------- 2. Prepare constructor args ----------
    default_args = dict(
        vocab_size=vocab_size,
        embed_dim=C.EMBED_DIM,
        num_classes=num_classes,
        kernel_sizes=C.KERNEL_SIZES,
        num_filters=C.NUM_FILTERS,
        hidden_dim=C.HIDDEN_DIM,
        num_layers=C.NUM_LAYERS,
        bidirectional=C.BIDIRECTIONAL,
        dropout=C.DROPOUT,
        padding_idx=C.PAD_IDX,
    )
    default_args.update(model_kwargs)  # checkpoint overrides defaults

    model_cls = MODEL_REGISTRY[arch_name]
    sig = inspect.signature(model_cls)
    init_args = {k: v for k, v in default_args.items() if k in sig.parameters}

    # ---------- 3. Build, load, move ----------
    model = model_cls(**init_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _predict_rows(
    rows: List[str],
    model: torch.nn.Module,
    vocab,
    device: torch.device,
    batch_size: int = 512,
) -> List[int]:
    """Vectorised inference over *rows* returning predicted label IDs."""
    encoded = [encode_sample(t, vocab, C.SEQ_LEN) for t in rows]
    tensor = torch.tensor(encoded, dtype=torch.long, device=device)

    preds: List[int] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(tensor), batch_size),
            desc="⏩  Inference",
            unit_scale=batch_size,
            unit="rows",
        ):
            batch = tensor[start : start + batch_size]
            logits = model(batch)
            preds.extend(logits.argmax(1).tolist())
    return preds


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(args: argparse.Namespace) -> None:
    device = device_from_config()
    if device.type == "cpu":
        warn_cpu_fallback()

    vocab = load_vocab(C.VOCAB_PATH)
    label2id = json.loads(Path(C.LABEL2ID_PATH).read_text(encoding="utf-8"))
    id2label = {v: k for k, v in label2id.items()}

    model = _load_model(len(vocab["stoi"]), len(id2label), device)

    # --------------------  Read input  -------------------- #
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "wbs_name" not in rdr.fieldnames:
            raise ValueError("CSV must contain a 'wbs_name' column")
        rows = [row["wbs_name"] for row in rdr]

    # --------------------  Predict ------------------------ #
    preds = _predict_rows(rows, model, vocab, device, batch_size=args.batch_size)

    # --------------------  Write results ------------------ #
    out_path = csv_path.with_name(csv_path.stem + "_predictions.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["wbs_name", "prediction"])
        for text, pred in zip(rows, preds):
            writer.writerow([text, id2label[pred]])

    print(f"✅ Predictions written → {out_path}\n")


# --------------------------------------------------------------------------- #
# CLI entry‑point                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with the best WBS‑Classifier checkpoint on a CSV "
            "containing a 'wbs_name' column."
        )
    )
    parser.add_argument("csv", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Mini‑batch size used during inference (default: 512).",
    )
    main(parser.parse_args())

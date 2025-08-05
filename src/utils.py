
"""
utils.py – Core helper utilities for WBS‑Classifier
==================================================
• RNG seeding
• Metric helpers: accuracy, macro‑F1
• Confusion‑matrix computation + plotting
• Lightweight vocab + tokenisation helpers
• Device fallback warnings
• Training‑curve plotting

Every public symbol is exported via ``__all__`` so callers can simply do ::

    from src.utils import accuracy, f1_macro, save_plot_curves

and IDEs will autocomplete only the sanctioned helpers.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import json
import random
import re

import matplotlib.pyplot as plt  # obey ChatGPT rules: no custom colours
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
    f1_score as _sk_f1,
)

# Lazy import to avoid hard dependency – will use if installed
try:
    from torchmetrics.functional import f1_score as _tm_f1_score  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dep
    _tm_f1_score = None  # type: ignore

# --------------------------------------------------------------------------- #
# Project‑wide config                                                         #
# --------------------------------------------------------------------------- #
from src.config import Config as C  # noqa: E402  (placed after top‑level deps)

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

__all__ = [
    "set_seed",
    "device_from_config",
    "warn_cpu_fallback",
    "accuracy",
    "f1_macro",
    "compute_confusion",
    "save_confusion_matrix",
    "save_plot_curves",
    "load_vocab",
    "encode_sample",
    "tokenise",
]

# --------------------------------------------------------------------------- #
# RNG control                                                                 #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:  # noqa: D401 – imperative style OK
    """Deterministic Python/NumPy/PyTorch RNG for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Device helpers                                                              #
# --------------------------------------------------------------------------- #

def device_from_config() -> torch.device:  # noqa: C901 – function is simple enough
    """Return a :class:`torch.device` honouring :pydata:`Config.DEVICE`.

    * ``"cpu"`` → always CPU.
    * ``"cuda"`` or ``"cuda:<idx>"`` → that GPU if available, else CPU fallback.
    * any other value (including ``"auto"``—the default) → automatic detection
      (CUDA if present, otherwise CPU).
    """
    requested = getattr(C, "DEVICE", "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)  # honour index if one was provided
        return torch.device("cpu")

    # "auto" or any unrecognised string → best available
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_CPU_BANNER = (
    "\n⚠️  Training on *CPU* – this will be **slow**.\n"
    "   Check that CUDA is installed and that you installed the *GPU* PyTorch wheel\n"
    "   (e.g. torch==2.x+cu118).\n"
)


def warn_cpu_fallback() -> None:
    """Pretty‑print a warning if we fell back to CPU despite C.DEVICE = 'cuda'."""
    if C.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print(_CPU_BANNER, flush=True)


# --------------------------------------------------------------------------- #
# Metric helpers                                                              #
# --------------------------------------------------------------------------- #

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Per‑row accuracy (mean of *exact* matches)."""
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    return float((preds == targets.cpu().numpy()).mean())


# Choose backend at import‑time for zero overhead in hot loops

def _torchmetrics_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro-averaged F1 using TorchMetrics (preferred when installed).

    * Do **not** rely on a global `Config.NUM_CLASSES`—derive the class count
      directly from the incoming tensor to avoid AttributeErrors when the
      config is refactored.
    """
    num_classes = logits.size(1)               # robust ⇒ always correct
    return float(
        _tm_f1_score(                          # type: ignore
            logits,
            targets,
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
    )


def _sklearn_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    return float(_sk_f1(targets.cpu().numpy(), preds, average="macro"))


f1_macro = _torchmetrics_f1 if _tm_f1_score is not None else _sklearn_f1  # type: ignore


# --------------------------------------------------------------------------- #
# Confusion‑matrix helpers                                                    #
# --------------------------------------------------------------------------- #

def compute_confusion(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:  # noqa: D401
    """Return raw *counts* confusion matrix (no normalisation)."""
    preds = logits.argmax(dim=1).cpu().numpy()
    return _sk_confusion_matrix(targets.cpu().numpy(), preds)


def save_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path) -> None:
    """Save a *visually clear* confusion‑matrix heatmap to *path*."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_plot_curves(history: Dict[str, List[float]], path: Path) -> None:
    """Save loss/metric curves stored in *history* to *path* (PNG)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, vals in history.items():
        ax.plot(vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score / loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Vocab + tokenisation helpers                                                #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenise(text: str) -> List[str]:
    """Lower‑case alphanumeric tokeniser used *everywhere* (train & inference)."""
    return _TOKEN_RE.findall(text.lower())


def load_vocab(path: Path | str):  # -> Dict[str, Dict[str, int]]
    """Load the serialised vocabulary produced by ``preprocess.py``."""
    return torch.load(Path(path))


def encode_sample(text: str, vocab: Dict[str, Dict[str, int]], seq_len: int) -> List[int]:
    """Convert *text* into a *padded* list of token ids using *vocab*."""
    stoi = vocab["stoi"] if "stoi" in vocab else vocab  # support either format
    pad_idx = stoi.get("<pad>", 0)
    unk_idx = stoi.get("<unk>", 1)

    ids = [stoi.get(tok, unk_idx) for tok in tokenise(text)][:seq_len]
    ids.extend([pad_idx] * (seq_len - len(ids)))
    return ids

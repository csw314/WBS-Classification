
"""
src/train.py
============

Single‑GPU/CPU training script for the WBS text‑classification task.
Works with any architecture registered in `MODEL_REGISTRY` (`text_cnn`, `gru_rnn`, `lstm_rnn`)
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from time import time
from typing import List
import json
import datetime as _dt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config as C
from src.data.dataset import WBSDataset, wbs_collate
from src.models import MODEL_REGISTRY
from src.utils import (
    accuracy,
    f1_macro,
    set_seed,
    warn_cpu_fallback,
    save_plot_curves,
    device_from_config,
)

__all__: List[str] = ["main"]

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def build_model(vocab_size: int, num_classes: int) -> torch.nn.Module:
    """
    Instantiate the architecture requested in Config.MODEL_ARCH,
    passing *only* the hyper-parameters that the class actually accepts.
    """
    model_cls = MODEL_REGISTRY[C.MODEL_ARCH]

    # Candidate args from Config (cover everything we might need)
    candidate = dict(
        vocab_size   = vocab_size,
        embed_dim    = C.EMBED_DIM,
        num_classes  = num_classes,
        # CNN-specific ------------
        kernel_sizes = C.KERNEL_SIZES,
        num_filters  = C.NUM_FILTERS,
        # RNN-specific ------------
        hidden_dim   = C.HIDDEN_DIM,
        num_layers   = C.NUM_LAYERS,
        bidirectional= C.BIDIRECTIONAL,
        # Shared ------------------
        dropout      = C.DROPOUT,
        padding_idx  = C.PAD_IDX,
    )

    # Keep only parameters declared in the model’s __init__ signature
    sig = inspect.signature(model_cls)
    init_args = {k: v for k, v in candidate.items() if k in sig.parameters}

    return model_cls(**init_args)

def epoch_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer | None,
    device: torch.device,
    max_grad_norm: float | None = None,
):
    """Run *one* epoch. Returns **avg loss**, **accuracy**, and **macro‑F1**."""
    is_train = optimiser is not None
    model.train(mode=is_train)

    total_loss = 0.0
    n_batches = 0

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(logits, labels)

        if is_train:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimiser.step()

        # Aggregate for epoch‑level metrics
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    
    # Concatenate along batch dim
    logits_epoch = torch.cat(all_logits, dim=0)
    labels_epoch = torch.cat(all_labels, dim=0)

    avg_loss = total_loss / n_batches
    acc_epoch = accuracy(logits_epoch, labels_epoch)
    f1_epoch = f1_macro(logits_epoch, labels_epoch)

    return avg_loss, acc_epoch, f1_epoch

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    set_seed(C.SEED)

    # --------------------  Device ----------------------------------------- #
    device = device_from_config()
    if device.type == "cpu":
        warn_cpu_fallback()

    # --------------------  Data ------------------------------------------- #
    train_ds = WBSDataset(split="train")
    val_ds = WBSDataset(split="val")

    NUM_WORKERS = 0 if os.name == "nt" else 4  # Windows cannot fork safely
    
    train_loader = DataLoader(
        train_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        collate_fn=wbs_collate,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        collate_fn=wbs_collate,
        num_workers=NUM_WORKERS,
    )

    # --------------------  Model / Optim ---------------------------------- #
    model = build_model(train_ds.vocab_size, train_ds.num_classes).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=C.LR,
        weight_decay=getattr(C, "WEIGHT_DECAY", 0.0),
    )

    # --------------------  Gradient Clipping & early-stop params ---------- #
    max_grad_norm = getattr(C, "MAX_GRAD_NORM", None)
    patience = getattr(C, "EARLY_STOPPING_PATIENCE", 4)
    min_delta = getattr(C, "EARLY_STOPPING_MIN_DELTA", 0.0)

    best_val_loss = math.inf
    epochs_without_improve = 0

    history: dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    # --------------------  Train loop ------------------------------------- #
    for epoch in range(1, C.EPOCHS + 1):
        start = time()
        tr_loss, tr_acc, tr_f1 = epoch_loop(model, train_loader, optimiser, device, max_grad_norm)
        vl_loss, vl_acc, vl_f1 = epoch_loop(model, val_loader, None, device)
        dur = time() - start

        # --------------------  Plots / log ------------------------------------ #
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        print(
            f"Epoch {epoch:3d}/{C.EPOCHS} • {dur:4.1f}s\n"
            f" train loss {tr_loss:6.4f}  acc {tr_acc:6.4f}  f1 {tr_f1:6.4f}\n"
            f" val   loss {vl_loss:6.4f}  acc {vl_acc:6.4f}  f1 {vl_f1:6.4f}",
            flush=True,
        )

        # ---------------- Early‑stopping + checkpoint -------------------- #
        if vl_loss + min_delta < best_val_loss:
            best_val_loss = vl_loss
            epochs_without_improve = 0

            # Build minimal kwargs to reconstruct the model during inference
            model_cls = MODEL_REGISTRY[C.MODEL_ARCH]
            sig = inspect.signature(model_cls)
            candidate = dict(
                vocab_size=model.vocab_size,
                embed_dim=C.EMBED_DIM,
                num_classes=model.num_classes,
                kernel_sizes=C.KERNEL_SIZES,
                num_filters=C.NUM_FILTERS,
                hidden_dim=C.HIDDEN_DIM,
                num_layers=C.NUM_LAYERS,
                bidirectional=C.BIDIRECTIONAL,
                padding_idx=C.PAD_IDX,
                dropout=C.DROPOUT,
                model_arch=C.MODEL_ARCH,
            )
            model_kwargs = {k: v for k, v in candidate.items() if k in sig.parameters or k == "model_arch"}

            torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs}, C.BEST_MODEL_PATH)
            print(f"🚀  New best model saved → {C.BEST_MODEL_PATH}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(
                    f"Early stopping: no val‑loss improvement for {patience} epochs.",
                    flush=True,
                )
                break
    # ---------------------  Final Plots ------------------------------------ #
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_plot_curves(history, C.FIG_DIR / "training_curves.png")

    # ------------------------------------------------------------------ #
    # Save raw history as JSON so notebooks & dashboards can reuse it
    # ------------------------------------------------------------------ #
    json_path = C.FIG_DIR / f"training_curves_{C.MODEL_ARCH}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)          # safety
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "epoch":       list(range(1, len(history["train_loss"]) + 1)),
                "train_loss":  history["train_loss"],
                "val_loss":    history["val_loss"],
                "train_acc":   history["train_acc"],
                "val_acc":     history["val_acc"],
                "train_f1":    history["train_f1"],
                "val_f1":      history["val_f1"],
            },
            fh,
            indent=2,
        )
    print(f"📈  Metrics JSON saved to {json_path}")


# --------------------------------------------------------------------------- #
# CLI entry‑point                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Train a WBS Classifier on the processed dataset."
    )

    # --- Core model selection ------------------------------------------------
    parser.add_argument(
        "--model_arch",
        choices=MODEL_REGISTRY.keys(),
        default=C.MODEL_ARCH,
        help="Model architecture to use (overrides Config.MODEL_ARCH).",
    )

    # --- Frequent hyper‑params ----------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=C.EPOCHS,
        help=f"Number of training epochs (default: {C.EPOCHS}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=C.BATCH_SIZE,
        help=f"Mini‑batch size (default: {C.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=C.LR,
        help=f"Learning rate for AdamW (default: {C.LR}).",
    )

    # --- Runtime settings ----------------------------------------------------
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=C.DEVICE,
        help="Device to train on: 'cpu', 'cuda', or 'auto' to choose automatically.",
    )

    args = parser.parse_args()

    # ---- Propagate CLI overrides into the Config singleton ------------------
    C.MODEL_ARCH = args.model_arch
    C.EPOCHS = args.epochs
    C.BATCH_SIZE = args.batch_size
    C.LR = args.lr
    C.DEVICE = args.device

    # Kick off training -------------------------------------------------------
    main(args)


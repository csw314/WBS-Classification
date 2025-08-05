
"""src/evaluate.py
====================
Comprehensive evaluation & data‑diagnostics script for the WBS Text‑CNN.

Run from the project root, for example:

```bash
python -m src.evaluate --split val \
                       --batch_size 256 \
                       --outdir reports/val_eval
```

Outputs into `--outdir`:
* **predictions.csv** – per‑row ground‑truth vs predicted class (+ project_id & optional WBS text)
* **classification_report.json** – sklearn precision / recall / F1 per class
* **confusion_matrix.png** – heat‑map of class‑confusion counts
* **length_bucket_accuracy.csv** – accuracy across token‑length buckets
* **data_distribution.csv** – class counts for *train / val / test* splits

The script prints a short summary to stdout and saves artefacts for offline
error‑analysis.  All plotting uses matplotlib (no seaborn per project rules).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import inspect 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import Config as C
from src.data.dataset import WBSDataset, wbs_collate
from src.models import MODEL_REGISTRY
from src.utils import device_from_config, load_vocab

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model & dump diagnostics.")
    parser.add_argument("--model_arch", choices=MODEL_REGISTRY.keys(), default=C.MODEL_ARCH, 
                        help="Override the architecture defined in src.config.Config")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: val)")
    parser.add_argument("--batch_size", type=int, default=256, help="Eval batch size")
    parser.add_argument("--outdir", default="reports/eval", help="Directory to write outputs")
    parser.add_argument("--with_text", action="store_true",
                        help="Include original wbs_name text in predictions.csv (requires raw CSV access)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def to_cpu(t):
    return t.detach().cpu() if torch.is_tensor(t) else t

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Evaluation proper
# ---------------------------------------------------------------------------


def evaluate(split: str, batch_size: int, outdir: Path, with_text: bool = False) -> None:
    device = device_from_config()

    # ----- Dataset & Loader --------------------------------------------------
    ds = WBSDataset(split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=wbs_collate)

    # ----- Vocab & labels ----------------------------------------------------
    vocab = load_vocab(C.VOCAB_PATH)
    label2id: Dict[str, int] = json.loads(Path(C.LABEL2ID_PATH).read_text("utf-8"))
    id2label: Dict[int, str] = {v: k for k, v in label2id.items()}

    # ----- Model -------------------------------------------------------------
    ckpt = torch.load(C.BEST_MODEL_PATH, map_location=device)

    # 1) Detect old-vs-new checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict   = ckpt["state_dict"]
        model_kwargs = ckpt.get("model_kwargs", {})          # may be empty
        arch_name    = model_kwargs.pop("model_arch", C.MODEL_ARCH)
    else:  # legacy weights-only checkpoint
        state_dict   = ckpt
        model_kwargs = {}
        arch_name    = C.MODEL_ARCH

    # 2) Fill in any missing, *generic* defaults -------------
    generic_defaults = dict(
        vocab_size=len(vocab["stoi"]),
        embed_dim=C.EMBED_DIM,
        num_classes=len(id2label),
        kernel_sizes=getattr(C, "KERNEL_SIZES", (3, 4, 5)),
        num_filters=getattr(C, "NUM_FILTERS", 100),
        hidden_dim=getattr(C, "HIDDEN_DIM", 256),
        num_layers=getattr(C, "NUM_LAYERS", 1),
        bidirectional=getattr(C, "BIDIRECTIONAL", False),
        dropout=C.DROPOUT,
        padding_idx=C.PAD_IDX,
    )
    generic_defaults.update(model_kwargs)  # kwargs from ckpt win

    # 3) Keep only parameters accepted by the chosen class --
    model_cls = MODEL_REGISTRY[arch_name]
    sig       = inspect.signature(model_cls)
    init_args = {k: v for k, v in generic_defaults.items() if k in sig.parameters}

    model = model_cls(**init_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ----- Inference loop ----------------------------------------------------
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device))
            y_pred.extend(to_cpu(logits.argmax(1)).tolist())
            y_true.extend(to_cpu(batch["labels"]).tolist())

    # -----------------------------------------------------------------------
    # 1. predictions.csv -----------------------------------------------------
    # -----------------------------------------------------------------------
    df = pd.DataFrame({
        "project_id": ds.project_id,
        "true_label": [id2label[i] for i in y_true],
        "pred_label": [id2label[i] for i in y_pred],
    })

    if with_text:
        try:
            raw = pd.read_csv(C.RAW_CSV)
            raw = raw[raw["level_1"].str.lower() != "unmapped"].reset_index(drop=True)
            proj_ids_for_split = json.loads(Path(C.PROJECT_SPLITS[split]).read_text("utf-8"))
            raw_subset = raw[raw["project_id"].isin(proj_ids_for_split)].reset_index(drop=True)
            if len(raw_subset) == len(df):
                df["wbs_name"] = raw_subset["wbs_name"].values
            else:
                print("⚠️  Could not align raw CSV rows; skipping wbs_name")
        except Exception as exc:
            print(f"⚠️  Failed to merge wbs_name text: {exc}")

    ensure_dir(outdir)
    pred_path = outdir / f"{split}_predictions.csv"
    df.to_csv(pred_path, index=False)

    # -----------------------------------------------------------------------
    # 2. Classification report ----------------------------------------------
    # -----------------------------------------------------------------------
    rep = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))], output_dict=True)
    (outdir / "classification_report.json").write_text(json.dumps(rep, indent=2))

    # -----------------------------------------------------------------------
    # 3. Confusion matrix heat‑map -------------------------------------------
    # -----------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix – split: %s" % split)
    fig.colorbar(im)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(id2label)))
    ax.set_yticks(range(len(id2label)))
    ax.set_xticklabels([l[:10] for l in id2label.values()], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([l[:10] for l in id2label.values()], fontsize=8)
    plt.tight_layout()
    fig_path = outdir / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 4. Length‑bucket accuracy ---------------------------------------------
    # -----------------------------------------------------------------------
    # Non‑PAD token count per sample (input_ids == PAD_IDX)
    pad_idx = vocab["stoi"]["<pad>"]
    lengths = (ds.input_ids != pad_idx).sum(1).tolist()

    buckets = [(0, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
    records = []
    for lo, hi in buckets:
        idxs = [i for i, l in enumerate(lengths) if lo <= l <= hi]
        if idxs:
            acc = np.mean([y_true[i] == y_pred[i] for i in idxs])
            records.append({"bucket": f"{lo}-{hi}", "count": len(idxs), "accuracy": acc})
    pd.DataFrame(records).to_csv(outdir / "length_bucket_accuracy.csv", index=False)

    # -----------------------------------------------------------------------
    # 5. Data distribution across splits ------------------------------------
    # -----------------------------------------------------------------------
    dist_records = []
    for sp in ("train", "val", "test"):
        d = WBSDataset(split=sp)
        cnt = Counter(d.labels.tolist())
        for cls_id, n in cnt.items():
            dist_records.append({"split": sp, "label": id2label[cls_id], "count": n})
    pd.DataFrame(dist_records).to_csv(outdir / "data_distribution.csv", index=False)

    # -----------------------------------------------------------------------
    # Console summary --------------------------------------------------------
    # -----------------------------------------------------------------------
    print("\n=== Evaluation summary (split: %s) ===" % split)
    print("Accuracy: %.4f" % rep["accuracy"])
    print("Macro F1: %.4f" % rep["macro avg"]["f1-score"])
    print("Artifacts written to", outdir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    evaluate(split=args.split, batch_size=args.batch_size, outdir=Path(args.outdir), with_text=args.with_text)

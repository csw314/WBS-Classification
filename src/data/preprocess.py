
"""
preprocess.py – Create project‑level train/val/test splits, build the
vocabulary, and serialise padded/encoded tensors for the WBS‑Classifier
pipeline.

Updates (2025‑07‑29)
--------------------
* Replaced the local *whitespace* tokeniser with the **project‑wide**
  ``src.utils.tokenise`` helper so training and inference now share exactly
  the same rules.
* Removed the redundant ``def tokenize`` function.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.config import Config as C
from src.utils import tokenise  # 🔄 unified tokeniser

SEED = 42
random.seed(SEED)

# --------------------------------------------------------------------------- #
# CLI helpers                                                                 #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse CLI arguments; fall back to sensible defaults from ``Config``."""
    p = argparse.ArgumentParser(
        description="Preprocess raw WBS data into tensors & vocab for modelling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=C.RAW_CSV, help="Raw CSV file")
    p.add_argument("--out-dir", type=Path, default=C.PROC_DIR, help="Output directory for processed artefacts")
    p.add_argument("--val-split", type=float, default=0.1, help="Fraction of *projects* used for validation")
    p.add_argument("--test-split", type=float, default=0.1, help="Fraction of *projects* used for testing")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Splitting logic                                                             #
# --------------------------------------------------------------------------- #

def dominant_label(series: pd.Series) -> str:
    """Return the most common label in ``series`` (ties → first encountered)."""
    return series.value_counts().idxmax()


def split_projects(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split *projects* into train/val/test.

    Projects whose **dominant label** occurs in < 2 projects are forced into
    the *train* split so scikit-learn’s stratifier is happy.
    """
    # One row per project with its dominant label
    proj_tbl = (
        df.groupby("project_id", as_index=False)["level_1"]
          .agg(dominant_label)
          .rename(columns={"level_1": "label"})
    )

    # Identify rare-label projects (< 2 projects in that label)
    label_counts = proj_tbl["label"].value_counts()
    rare_labels = label_counts[label_counts < 2].index
    common_projects = proj_tbl[~proj_tbl["label"].isin(rare_labels)]
    rare_projects   = proj_tbl[proj_tbl["label"].isin(rare_labels)]

    # First split common projects → tmp (train+val) vs test
    proj_tmp, proj_test = train_test_split(
        common_projects,
        test_size=test_frac,
        stratify=common_projects["label"],
        random_state=seed,
    )

    # Now split tmp → train vs val
    val_adj = val_frac / (1.0 - test_frac)  # val fraction *within* tmp
    proj_train, proj_val = train_test_split(
        proj_tmp,
        test_size=val_adj,
        stratify=proj_tmp["label"],
        random_state=seed,
    )

    # Append rare-label projects to *train*
    proj_train = pd.concat([proj_train, rare_projects], ignore_index=True)

    return (
        proj_train["project_id"].tolist(),
        proj_val["project_id"].tolist(),
        proj_test["project_id"].tolist(),
    )

# --------------------------------------------------------------------------- #
# Vocab building                                                              #
# --------------------------------------------------------------------------- #

def build_vocab(token_lists: Sequence[List[str]]) -> Dict[str, Dict[str, int]]:
    """Return *stoi & itos* dicts with reserved ``<pad>``/``<unk>`` tokens."""
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    # Reserve indices 0 & 1
    stoi = {"<pad>": 0, "<unk>": 1}
    itos = ["<pad>", "<unk>"]

    for tok, _ in counter.most_common():
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)

    return {"stoi": stoi, "itos": itos}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:  # noqa: C901 – long but linear flow
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Reading raw CSV …", flush=True)
    df = pd.read_csv(args.csv, dtype={"project_id": str})

    # 1. Drop unmapped rows -------------------------------------------------- #
    df = df[df["level_1"] != "Unmapped"].reset_index(drop=True)

    # 2. Project‑level split ------------------------------------------------- #
    print("🔀 Stratified project split …", flush=True)
    train_ids, val_ids, test_ids = split_projects(
        df, val_frac=args.val_split, test_frac=args.test_split, seed=SEED
    )
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)

    # 3. Tokenise & vocab ---------------------------------------------------- #
    print("📝 Tokenising & building vocab …", flush=True)
    df["tokens"] = [tokenise(t) for t in tqdm(df["wbs_name"], leave=False)]
    train_tokens = df[df["project_id"].isin(split_map["train"])]
    vocab = build_vocab(train_tokens["tokens"].tolist())
    torch.save(vocab, out_dir / "vocab.pt")

    # 4. Encode & save tensors --------------------------------------------- #
    print("📦 Encoding tensors …", flush=True)
    stoi = vocab["stoi"]
    pad_idx = stoi["<pad>"]
    unk_idx = stoi["<unk>"]

    def encode(tok_list: List[str]) -> torch.Tensor:
        ids = [stoi.get(tok, unk_idx) for tok in tok_list][: C.SEQ_LEN]
        ids.extend([pad_idx] * (C.SEQ_LEN - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    encoded = torch.stack([encode(t) for t in tqdm(df["tokens"], leave=False)])
    torch.save(encoded, out_dir / "all_wbs.pt")

    # 5. Save label mapping -------------------------------------------------- #
    label2id = {lbl: i for i, lbl in enumerate(sorted(df["level_1"].unique()))}
    id2label = {v: k for k, v in label2id.items()}
    with open(out_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(out_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    print("✅ Preprocessing complete!")


if __name__ == "__main__":  # pragma: no cover
    main()



from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.config import Config as C


class WBSDataset(Dataset):
    """Simple in‑memory dataset for WBS classification.

    Each item is a dict with:
    • input_ids: LongTensor[max_seq_len]
    • label:      int
    • project_id: str (kept as python‑string)
    """

    def __init__(self, split: str = "train"):
        _SPLIT_TO_FILE = {
            "train": C.TRAIN_TENSOR,
            "val":   C.VAL_TENSOR,
            "test":  C.TEST_TENSOR,
        }
        file_path = _SPLIT_TO_FILE[split.lower()]

        blob: dict = torch.load(file_path)          # ← still a dict

        # ---- correct field assignment ----
        self.input_ids = blob["input_ids"]          # LongTensor [N, seq_len]
        self.labels     = blob["labels"]            # LongTensor [N]
        self.project_id = blob["project_id"]        # list[str]

        # ---------------------------------------------------------------------
        #  Metadata handy for model-building
        # ---------------------------------------------------------------------
        self.num_classes = int(self.labels.max().item()) + 1
        self.vocab_size  = len(torch.load(C.VOCAB_PATH)["stoi"])

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "label":      int(self.labels[idx]),
            "project_id": self.project_id[idx],
        }


# ---------- Collate fn to keep project_id as list -------------------------

def wbs_collate(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)  # already padded
    labels    = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    project_ids = [item["project_id"] for item in batch]
    return {"input_ids": input_ids, "labels": labels, "project_id": project_ids}
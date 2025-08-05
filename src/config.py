
#!/usr/bin/env python
"""
config.py – Centralised hyper‑parameters *and* canonical path definitions
========================================================================
Import as::

    from src.config import Config as C

All values are class‑level constants so nothing needs instantiation.  This
keeps the rest of the code‑base tidy (no need to pass half a dozen arguments
into every function).

Feel free to tweak numbers and paths here; everything else will pick the
change up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

class Config:  # pylint: disable=too-few-public-methods
    """Static container for project settings.

    We *deliberately* avoid any dynamic initialisation (no `__init__`) so that
    a missing environment variable or unavailable GPU never prevents the
    module from importing.  Runtime checks (e.g. CUDA availability) are done
    elsewhere in `src.utils`.
    """

    # --------------------------------------------------------------------- #
    # 📂  Directory layout                                                  #
    # --------------------------------------------------------------------- #
    ROOT: Path = Path(__file__).resolve().parent.parent
    """Top‑level project folder (i.e. the one containing *src/* and *data/*)."""

    DATA_DIR: Path = ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROC_DIR: Path = DATA_DIR / "processed"

    # Raw file(s)
    RAW_CSV: Path = RAW_DIR / "wbs_data.csv"

    # Processed tensors + helper artefacts
    TRAIN_TENSOR: Path = PROC_DIR / "train.pt"
    VAL_TENSOR: Path = PROC_DIR / "val.pt"
    TEST_TENSOR: Path = PROC_DIR / "test.pt"

    VOCAB_PATH: Path = PROC_DIR / "vocab.pt"
    LABEL2ID_PATH: Path = PROC_DIR / "label2id.json"

    PROJECT_SPLITS: Dict[str, Path] = {
        "train": PROC_DIR / "projects_train.json",
        "val": PROC_DIR / "projects_val.json",
        "test": PROC_DIR / "projects_test.json",
    }

    BEST_MODEL_PATH: Path = ROOT / "best_model.pth"

    FIG_DIR: Path = ROOT / "reports" / "figures"

    # --------------------------------------------------------------------- #
    # 🗄️  Data‑processing parameters                                        #
    # --------------------------------------------------------------------- #
    SEQ_LEN: int = 64              # tokens 
    MAX_VOCAB: int = 20_000        # most common tokens kept
    PAD_TOKEN: str = "<pad>"
    UNK_TOKEN: str = "<unk>"

    # The index of the PAD token will be set during preprocessing and saved
    # alongside the vocab, but default to 0 so the model can be constructed
    # *before* the vocab is loaded (e.g. when scripting).
    PAD_IDX: int = 0

    # --------------------------------------------------------------------- #
    # 🔀  Dataset split ratios & random seed                                 #
    # --------------------------------------------------------------------- #
    VAL_SPLIT: float = 0.10          # 10 % of *projects* → hold-out cross-validation
    TEST_SPLIT: float = 0.10         # 10 % of *projects* → hold‑out test
    SEED: int = 42                   # ensures reproducibility

    # --------------------------------------------------------------------- #
    # 🧠  Model hyper‑parameters                                            #
    # --------------------------------------------------------------------- #
    MODEL_ARCH: str = "lstm_rnn"      # looked‑up through MODEL_REGISTRY
    HIDDEN_DIM: int = 256
    NUM_LAYERS: int = 2
    BIDIRECTIONAL: bool = True
    EMBED_DIM: int = 128
    NUM_FILTERS: int = 100
    KERNEL_SIZES: Tuple[int, int, int] = (3, 4, 5)
    DROPOUT: float = 0.5
    WEIGHT_DECAY: float = 0.0001            # L2 reg
    MAX_GRAD_NORM: float = 2.0              # Clip Threshold
    EARLY_STOPPING_PATIENCE: int = 4        # Epochs
    EARLY_STOPPING_MIN_DELTA: float = 0.002  

    # --------------------------------------------------------------------- #
    # 🏋🏻‍♂️  Training hyper‑parameters                                      #
    # --------------------------------------------------------------------- #
    BATCH_SIZE: int = 128
    EPOCHS: int = 15
    LR: float = 0.0001

    # --------------------------------------------------------------------- #
    # ⚙️  Runtime / device settings                                         #
    # --------------------------------------------------------------------- #
    DEVICE: str = "cuda"  # will fall back to CPU in utils.device_from_config()

    # --------------------------------------------------------------------- #
    # 📜  Misc                                                               #
    # --------------------------------------------------------------------- #
    LOG_INTERVAL: int = 50  # batches between progress‑bar updates

    # Derived/alias properties 
    @classmethod
    def project_tensor(cls, split: str) -> Path:  # pragma: no cover
        """Return the tensor file corresponding to *split* (train/val/test)."""
        return {
            "train": cls.TRAIN_TENSOR,
            "val": cls.VAL_TENSOR,
            "test": cls.TEST_TENSOR,
        }[split]

    @classmethod
    def project_list(cls, split: str) -> Path:  # pragma: no cover
        """Return the JSON file containing the list of projects in *split*."""
        return cls.PROJECT_SPLITS[split]

# Only echo the banner in the main process (DataLoader workers are child procs)
import os, multiprocessing as _mp 
if _mp.current_process().name == "MainProcess":     # one-time, readable
    print("✅ Configuration complete →", Config)

# A convenient alias so downstream modules can `from src.config import C`.
C = Config

__all__ = [
    "Config",
    "C",
]

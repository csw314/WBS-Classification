# WBS Classifier

A **Work‑Breakdown‑Structure (WBS) text classifier** that sorts cost‑line descriptions into one of eight high‑level categories:

| ID | Category                                 |
| -- | ---------------------------------------- |
| 0  | Design and NRE                           |
| 1  | OPC Testing & Startup                    |
| 2  | Construction                             |
| 3  | Process Equipment                        |
| 4  | D\&D (Decontamination & Decommissioning) |
| 5  | SEPM (System Engineering & Project Mgmt) |
| 6  | Other Labor & Support                    |
| 7  | Standard Equipment                       |

Built with **PyTorch 2.2**, the project provides an end‑to‑end pipeline—**pre‑processing → training → evaluation → inference**—and ships three ready‑made neural architectures (Text‑CNN, GRU, LSTM).  All paths and hyper‑parameters are centrally managed in `src/config.py` so you can experiment without touching the plumbing.

---

## 📑 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Project Layout](#project-layout)
4. [Configuration](#configuration)
5. [Extending / Contributing](#extending--contributing)
6. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Installation

```bash
# 1 ▪ Clone the repo
$ git clone https://github.com/your‑org/wbs_classifier.git
$ cd wbs_classifier

# 2 ▪ (Recommended) create a fresh virtual env
$ python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3 ▪ Install Python dependencies
$ pip install -r requirements.txt
```

The project has been tested on **Python 3.10+** with both CPU and CUDA 11.8 wheels of PyTorch.

---

## Quick Start

Below is the minimal happy‑path flow. All commands are run from the repository root.

### 1 ▪ Pre‑process raw data

Converts your raw CSV into token tensors, vocabulary, and train/val/test splits.

```bash
$ python -m src.data.preprocess \
        --csv data/raw/wbs_data.csv  # adjust path if needed
```

Generated artefacts land in `data/processed/` and are *git‑ignored*.

### 2 ▪ Train a model

```bash
$ python -m src.train                          # uses hyper‑params in config.py
# or override on the CLI
$ python -m src.train --model_arch text_cnn
```

The best checkpoint is saved to **`best_model.pth`** with early‑stopping.

### 3 ▪ Evaluate

```bash
$ python -m src.evaluate --split val --outdir reports/val_eval
```

You’ll get:

* `predictions.csv` – ground‑truth vs prediction per row
* `classification_report.json` – precision / recall / F1 per class
* `confusion_matrix.png`, `training_curves.png`, etc.

### 4 ▪ Run inference

```bash
$ python -m src.predict data/raw/new_wbs.csv
# → data/raw/new_wbs_predictions.csv
```

`new_wbs.csv` must contain a `wbs_name` column.

---

## Project Layout

```
wbs_classifier/
├── data/                   # raw & processed data (git‑ignored except tiny samples)
│   ├── raw/
│   └── processed/
├── reports/
│   └── exploratory.ipynb   # Jupyter exploration
├── reports/                # evaluation artefacts & plots
├── src/                    # all Python source code
│   ├── data/               # preprocess.py, dataset.py, collate helpers
│   ├── models/             # text_cnn.py, gru_rnn.py, lstm_rnn.py, registry
│   └── utils.py            # misc helpers (metrics, tokeniser, plotting)       
├── best_model.pth          # latest/best checkpoint (ignored by Git LFS)
├── README.md               # latest/best checkpoint (ignored by Git LFS)
└── Requirements.txt        # package versions
```

> **Tip:** Every script carries a detailed doc‑string and can be invoked with `‑h` for help.

---

## Configuration

All knobs live in **`src/config.py`**:

```python
class Config:
    MODEL_ARCH = "lstm_rnn"      # text_cnn | gru_rnn | lstm_rnn
    SEQ_LEN    = 64              # tokens per sample
    EMBED_DIM  = 128
    HIDDEN_DIM = 256
    LR         = 1e‑4
    EPOCHS     = 15
    BATCH_SIZE = 128
    DEVICE     = "cuda"          # "cpu" forces CPU, "auto" picks best
    ...
```

Changes are *single‑source‑of‑truth*: every script imports `Config` so you never duplicate paths or magic numbers.

---

## Extending / Contributing

* **Add a new model** → drop a module in `src/models/` and register it in `src/models/__init__.py` (see comments there).
* **Tune hyper‑parameters** → edit `config.py` or pass `--flag value` on the CLI.
* **Speed‑ups** → enable mixed‑precision, batch‑size sweeps, or multi‑GPU training (script skeletons provided in `scripts/experimental/`).

Please open a PR or issue if you hit problems—the maintainers welcome improvements!

---

## Troubleshooting & FAQ

| Symptom                             | Fix                                                                                                             |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| *Training is stuck on CPU*          | Make sure you installed the **CUDA** wheel of PyTorch (`+cu118`) and that `Config.DEVICE` is set to `"cuda"`.   |
| *`FileNotFoundError: wbs_data.csv`* | Place your data under `data/raw/` or pass `--csv path/to/file`.                                                 |
| *Low F1 score*                      | Check class imbalance in `reports/eval/data_distribution.csv`; try `--model_arch gru_rnn` or additional epochs. |

---

© 2025 Department of Energy – Internal prototype. Use at your own risk.

# HGOOD

**Hypergraph-enhanced Graph Out-of-Distribution Detection**

Official implementation of our work **accepted to [IJCAI-ECAI 2026](https://www.ijcai.org/)**.

> **Important:** The scripts in `script/` are the **original, un-tuned launch scripts** used during development. They are **not** the fully hyperparameter-tuned configurations reported in the paper. For reproducibility of the paper numbers, please refer to the paper appendix or contact the authors.


## Table of Contents
- [Publication](#publication)
- [Method Summary](#method-summary)
  - [Pipeline](#pipeline)
  - [Key Components](#key-components)
  - [Tasks](#tasks)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
  - [TU Graph Benchmarks](#tu-graph-benchmarks)
  - [OGB Molecular Graphs](#ogb-molecular-graphs)
  - [Tox21 Anomaly Detection](#tox21-anomaly-detection)
- [Quick Start](#quick-start)
- [Experiment Types](#experiment-types)
  - [OOD Detection](#ood-detection--exp_type-oodd)
  - [Anomaly Detection](#anomaly-detection--exp_type-ad)
- [Command-Line Arguments](#command-line-arguments)
- [Provided Scripts](#provided-scripts)
  - [OOD Detection Scripts](#ood-detection-1)
  - [Anomaly Detection Scripts](#anomaly-detection-1)
- [Implementation Details](#implementation-details)
  - [Training Loop](#training-loop-per-epoch)
  - [Random Seed](#random-seed)
  - [Final Metrics](#final-metrics)
- [Training Output](#training-output)

---

## Publication

This repository accompanies our paper accepted to **IJCAI-ECAI 2026**.

If you use this code in your research, please cite our paper (BibTeX to be updated upon publication) and, if applicable, GOOD-D:

```bibtex
@inproceedings{liu2023goodd,
  title={GOOD-D: On Unsupervised Graph Out-of-Distribution Detection},
  author={Liu, Yixin and Ding, Kaize and Liu, Huan and Pan, Shirui},
  booktitle={Proceedings of the 16th ACM International Conference on Web Search and Data Mining},
  pages={339--347},
  year={2023}
}
```

---

## Method Summary

### Pipeline

```
Input graph
  ├─ Feature view x           ──► GIN encoder (encoder_feat)
  └─ Structure view x_s (RW+DG) ──► GIN encoder (encoder_str)
                                    │
                                    ├─ Graph- / node-level contrast (loss_g, loss_n)
                                    └─ Prototype contrast (loss_b, FAISS K-Means)

Input graph (same)
  ├─ x  ──► HypergraphEncoder (encoder_hyper_feat)
  └─ x_s ──► HypergraphEncoder (encoder_hyper_str)
              │
              └─ Hypergraph branch graph embedding hyper_b

Cross-modal alignment
  └─ Bidirectional prototype contrast (loss_cross_modal)
       (GIN embeddings ↔ hypergraph prototypes, and vice versa)

Inference
  └─ Fuse y_score_b + y_score_g + y_score_n + y_score_cross_modal → AUROC
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Structural encoding** | Perturbation-free RWSE + degree one-hot encoding (`x_s`), inherited from GOOD-D |
| **GIN dual encoders** | Separate encoders for feature view and structure view |
| **Hypergraph encoder** | Learnable hyperedge prototypes + soft node–hyperedge assignment + hypergraph convolution |
| **Multi-level losses** | Node (`loss_n`), graph (`loss_g`), prototype (`loss_b`), cross-modal (`loss_cross_modal`) |
| **Adaptive weighting** | Loss weights adjusted by per-epoch standard deviation (`-alpha`) |
| **Scoring** | Prototype distance + cross-modal dissimilarity, aggregated at test time |

### Tasks

| Task | Flag | Setting |
|------|------|---------|
| Graph-level OOD detection | `-exp_type oodd` | Train on unlabeled ID graphs; test on mixed ID/OOD graphs |
| Graph-level anomaly detection | `-exp_type ad` | Train on normal (majority) graphs only; test includes anomalies |

Both tasks use **AUROC** as the evaluation metric.

---

## Repository Layout

```
HGOOD/                          ← Project root (this README)
├── README.md
├── .vscode/
│   └── settings.json
└── HGOOD/                      ← Main code (cd here before running)
    ├── main.py                 ← Training & evaluation entry point
    ├── model.py                ← HCL model, hypergraph encoder, losses, scoring
    ├── data_loader.py          ← Data loading, structural encoding, splits
    ├── Params.py               ← Legacy params (not used by main.py)
    ├── Utils/
    │   ├── Utils.py
    │   └── TimeLogger.py
    ├── script/                 ← Experiment launch scripts (original, un-tuned)
    │   ├── oodd_*.sh
    │   └── ad_*.sh
    └── data/                   ← Dataset cache
        ├── PTC_MR/
        ├── MUTAG/
        └── ...
```

---

## Requirements

Recommended environment (aligned with GOOD-D; adjust for your CUDA version):

| Package | Suggested |
|---------|-----------|
| Python | 3.8+ |
| PyTorch | 1.11+ |
| PyTorch Geometric | 2.0+ |
| faiss-cpu or faiss-gpu | latest stable |
| scikit-learn, numpy, scipy, networkx | recent versions |
| ogb | required for OGB molecular datasets |

Example setup:

```bash
conda create -n hgood python=3.9 -y
conda activate hgood

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric faiss-cpu scikit-learn ogb networkx scipy
```

If FAISS GPU clustering fails, `main.py` automatically falls back to CPU (`run_kmeans`).

---

## Data Preparation

### TU Graph Benchmarks

Place TU-format data under `HGOOD/data/<DATASET_NAME>/`. Some datasets (e.g., `PTC_MR`, `MUTAG`) are already included. PyG `TUDataset` can also auto-download on first run.

### OGB Molecular Graphs

Specify pairs such as `-DS_pair ogbg-molbbbp+ogbg-molbace`. Data is downloaded automatically to `HGOOD/data/` on first use.

Common OOD pairs (GOOD-D benchmark):

| ID | OOD |
|----|-----|
| BZR | COX2 |
| PTC_MR | MUTAG |
| AIDS | DHFR |
| ENZYMES | PROTEINS |
| IMDB-MULTI | IMDB-BINARY |
| ogbg-molbbbp | ogbg-molbace |
| ogbg-moltox21 | ogbg-molsider |
| ogbg-molfreesolv | ogbg-moltoxcast |
| ogbg-molesol | ogbg-molmuv |
| ogbg-molclintox | ogbg-mollipo |

### Tox21 Anomaly Detection

For `-DS Tox21_<target>`, prepare GOOD-D-style folders:

```
data/Tox21_<target>_training/
data/Tox21_<target>_testing/
```

Each folder should contain TU-format files (`*_A.txt`, `*_graph_labels.txt`, etc.).

---

## Quick Start

```bash
cd HGOOD/HGOOD

# OOD detection: PTC_MR (ID) + MUTAG (OOD)
python main.py -exp_type oodd -DS_pair PTC_MR+MUTAG -num_epoch 400 -num_cluster 5 -alpha 0.4

# Anomaly detection: AIDS
python main.py -exp_type ad -DS AIDS -num_epoch 20 -num_cluster 3 -alpha 1.0

# Or use a provided script (original, un-tuned settings)
bash script/oodd_PTC_MR+MUTAG.sh
bash script/ad_AIDS.sh
```

---

## Experiment Types

### OOD Detection (`-exp_type oodd`)

- 90% of ID graphs → unlabeled training set
- Remaining 10% ID + equal number of OOD graphs → test set
- Labels: `y=0` (ID), `y=1` (OOD)

```bash
# Option 1: dataset pair
python main.py -exp_type oodd -DS_pair BZR+COX2 ...

# Option 2: separate flags
python main.py -exp_type oodd -DS BZR -DS_ood COX2 ...
```

### Anomaly Detection (`-exp_type ad`)

- **TU datasets:** 5-fold stratified split; training keeps majority class only
- **Tox21:** official train/test split; training keeps normal (`y=1`) samples only
- Minority / anomalous class labeled as `y=1` at test time

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-exp_type` | `oodd` | `oodd` or `ad` |
| `-DS` | `PTC_MR` | Primary / ID dataset |
| `-DS_ood` | `MUTAG` | OOD dataset (when `DS_pair` is not set) |
| `-DS_pair` | `None` | e.g. `PTC_MR+MUTAG` |
| `-rw_dim` | `16` | Random-walk structural encoding dim |
| `-dg_dim` | `16` | Degree encoding dim |
| `-batch_size` | `128` | Training batch size |
| `-batch_size_test` | `9999` | Test batch size |
| `-lr` | `0.0001` | Adam learning rate |
| `-num_layer` / `-num_gc_layer` | `5` | GIN / hypergraph encoder layers |
| `-hidden_dim` | `16` | Hidden dimension |
| `-num_trial` | `2` | Number of repeated runs |
| `-num_epoch` | `170` | Training epochs |
| `-eval_freq` | `10` | Evaluation interval |
| `-is_adaptive` | `1` | Enable adaptive loss weighting |
| `-num_cluster` | `2` | FAISS K-Means clusters K |
| `-alpha` | `0.2` | Adaptive weighting exponent |
| `-cross_modal_weight` | `1.0` | Cross-modal loss weight |

---

## Provided Scripts

All scripts live in `HGOOD/script/` and should be run from `HGOOD/HGOOD/`.

> **Note:** These are **original launch scripts without hyperparameter tuning**. They reflect early development settings and may differ from the configurations used for the final results in the IJCAI-ECAI 2026 paper. Some scripts also contain typos (see [Known Issues](#known-issues)).

### OOD Detection

| Script | Dataset Pair | Key Settings |
|--------|--------------|--------------|
| `oodd_BZR+COX2.sh` | BZR + COX2 | epoch=400, K=2, α=0 |
| `oodd_PTC_MR+MUTAG.sh` | PTC_MR + MUTAG | epoch=400, K=5, α=0.4 |
| `oodd_AIDS+DHFR.sh` | AIDS + DHFR | epoch=400, K=10, α=0.2 |
| `oodd_ENZYMES+PROTEINS.sh` | ENZYMES + PROTEINS |  epoch=400, K=10, α=0.2 |
| `oodd_IMDB-M+IMDB-B.sh` | IMDB-MULTI + IMDB-BINARY | epoch=20, K=5, α=0.9 |
| `oodd_bbbp+bace.sh` | ogbg-molbbbp + ogbg-molbace | epoch=400, K=30, α=0.2 |
| `oodd_tox21+sider.sh` | ogbg-moltox21 + ogbg-molsider | epoch=160, K=5, α=0.2 |
| `oodd_freesolv+toxcast.sh` | ogbg-molfreesolv + ogbg-moltoxcast | epoch=30, K=2, α=0.6 |
| `oodd_sol+muv.sh` | ogbg-molesol + ogbg-molmuv | epoch=400, lr=5e-5, K=30, α=0.5 |
| `oodd_clintox+lipo.sh` | ogbg-molclintox + ogbg-mollipo | epoch=300, K=30, α=0.6 |

### Anomaly Detection

| Script | Dataset | Key Settings |
|--------|---------|--------------|
| `ad_AIDS.sh` | AIDS | epoch=20, K=3, α=1.0 |
| `ad_BZR.sh` | ⚠ actually ENZYMES | epoch=400, K=10, α=0.2 |
| `ad_COX2.sh` | COX2 | epoch=150, K=3, α=0.4 |
| `ad_DHFR.sh` | DHFR | epoch=60, K=2, α=0 |
| `ad_ENZYMES.sh` | ENZYMES | epoch=400, K=10, α=0.2 |
| `ad_DD.sh` | DD | batch=16, epoch=100, K=2, α=1.0 |
| `ad_NCI1.sh` | NCI1 | epoch=400, K=20, α=1.0 |
| `ad_IMDB-B.sh` | IMDB-BINARY | epoch=400, K=10, α=0.2 |
| `ad_REDDIT-B.sh` | REDDIT-BINARY | epoch=80, K=5, α=0.8 |
| `ad_COLLAB.sh` | COLLAB | batch=64, epoch=100, K=2, α=0.8 |
| `ad_PROTEINS_full.sh` | PROTEINS_full | epoch=20, K=2, α=0.2 |
| `ad_p53.sh`, `ad_HSE.sh`, `ad_MMP.sh`, `ad_PPAR-gamma.sh` | Tox21 targets | see script comments |

---

## Implementation Details

### Training Loop (per epoch)

1. Extract embeddings on the training set; run FAISS K-Means for GIN and hypergraph branches separately.
2. Forward pass per batch; compute `loss_b`, `loss_g`, `loss_n`, `loss_cross_modal`.
3. Backprop and update parameters.
4. Record loss standard deviations for adaptive weighting in the next epoch.
5. Every `eval_freq` epochs, evaluate AUROC on the test set and track the best model.

### Random Seed

Default: `setup_seed(1)`. Some script comments note better seeds (e.g. `seed: 2,3,4`); change the argument in `main.py` manually if needed.

### Final Metrics

After `num_trial` runs, the top-3 best AUROC values (or all if fewer than 3) are averaged:

```
[FINAL RESULT] AVG_AUC:0.xxxx+-0.xxxx
```


---

## Training Output

Example log:

```
================
Exp_type: oodd
DS: PTC_MR+MUTAG
num_features: 1
num_structural_encodings: 32
hidden_dim: 16
num_gc_layers: 5
cross_modal_weight: 1.0
================
[TRAIN] Epoch:010 | Loss:0.8234
[EVAL] Epoch: 010 | AUC:0.7123
[BEST] New best AUC: 0.7123
...
[RESULT] Trial: 00 | AUC:0.7456
[FINAL RESULT] AVG_AUC:0.7412+-0.0031
```

---


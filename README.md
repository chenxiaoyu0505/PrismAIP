# PrismAIP

English · [简体中文](README_CN.md)

A four-view predictor for **anti-inflammatory peptides (AIP)**. Each peptide is
scored by four independent models and the scores are combined by a learned
rank-average; nothing is trained end-to-end.

| view | what it looks at | model |
|---|---|---|
| `hand` | 694 handcrafted descriptors + 10 amphipathicity values | ExtraTrees (1,500 trees) |
| `gap4` | tolerant 4-mers (one position blanked per window) | XGBoost |
| `k5` | contiguous 5-mers | LightGBM |
| `tlora` | ESM-2 650M with a data-aligned low-rank adapter on q/k/v | ESM-2 + adapter + positional head |

Each view's probability is mapped to a percentile rank against its training
out-of-fold distribution; the four ranks are combined with weights fitted on
those out-of-fold scores (Caruana forward selection, 27 rounds), then mapped to
a probability by a Platt fit; the decision threshold is the out-of-fold
MCC-optimal value. Because the reference distributions come from training data
only, a peptide's score does not depend on the other rows in the file being
scored.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. A GPU is only needed for the `tlora` view (~1.5 GB of VRAM at
inference); pass `--device cpu` to run without one.

### The ESM-2 backbone (needed for both scoring and retraining)

The `tlora` view runs on `facebook/esm2_t33_650M_UR50D`. Its weights are **not**
in this repository — they are ~2.6 GB and are pulled from Hugging Face the first
time you score or retrain, so the first run needs network access.

To fetch them ahead of time (one-off, ~2.6 GB):

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
m='facebook/esm2_t33_650M_UR50D'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
```

Behind a mirror or on a slow link, set these before the command above:

```bash
export HF_ENDPOINT=https://hf-mirror.com    # any Hugging Face mirror
export HF_HOME=/data/hf_cache               # where the ~2.6 GB lives
```

Once cached, the package runs **fully offline** — no calls home, no downloads:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/predict.py --input my.csv
```

If the backbone cannot be loaded, `predict.py` fails loudly rather than silently
scoring with three views; pass `--views hand,gap4,k5 --device cpu` if you want
the three classical views only (no ESM, no GPU, no download).

## Score peptides

```bash
python scripts/predict.py --input example_input.csv
```

Input: a csv with a `sequence` column (extra columns are kept). Output: the same
rows plus one probability per view and the fused result:

| column | meaning |
|---|---|
| `p_hand`, `p_gap4`, `p_k5`, `p_tlora` | per-view probability |
| `prismaip_score` | fused rank score (the quantity the weights act on) |
| `prismaip_prob` | calibrated probability |
| `prismaip_call` | `AIP` / `non-AIP` at the recommended threshold |

Useful flags: `--views hand,gap4,k5` (skip the neural views), `--device cpu`,
`--out my_scores.csv`.

## Reported performance

Held-out test set (`data/test1.csv`: 817 peptides, 262 positive). Values are the
mean ± sd over three seeds (0/1/2); each seed fits a stratified 5-fold CV
ensemble on `data/train1.csv` and is evaluated once on this test set. All model
selection and calibration used training data only.

| view | AUROC | AUPR |
|---|---|---|
| hand | 0.783 ± 0.001 | 0.595 ± 0.001 |
| gap4 | 0.805 ± 0.003 | 0.621 ± 0.004 |
| k5 | 0.820 ± 0.003 | 0.662 ± 0.007 |
| tlora | 0.803 ± 0.021 | 0.601 ± 0.054 |
| **fused** | **0.8468 ± 0.0055** | **0.6796 ± 0.0111** |

`verify.py` re-scores `test1` from the shipped single-model weights and checks
the recorded reference values (tolerance 2e-3):

```bash
python scripts/verify.py
```

## What is in the box

```
data/            train1.csv, test1.csv, manifest.json (sha256 + fold definition)
weights/ml/      one model per classical view, fitted on all of train1 (.xz)
weights/tlora/   five fold deltas + the frozen SVD initialisation of A
weights/fusion.json, fusion_ecdf.npz, oof.npz
scripts/         predict.py (entry), verify.py, train_all.py, and the view code
```

## Retrain

```bash
python scripts/train_all.py               # ML weights, fusion, expected scores
```

The three classical views are refit on the whole `train1` set (this is the
shipped configuration). The fusion weights are always fitted on the
cross-validated out-of-fold scores in `weights/oof.npz`, never on test data.

The cross-validation protocol behind the reported numbers is
`StratifiedKFold(5, shuffle=True, random_state=42)` over `train1` rows in file
order, repeated for seeds 0/1/2; `train_folds()` in `scripts/ml_views.py`
reproduces the fold ensembles.

Retraining needs the ESM-2 backbone cached (see *Install* above) and takes
~3 min on a GPU.

## Notes

* Sequences are upper-cased and read as-is; non-standard residues are ignored by
  the k-mer views, and nothing is truncated for the classical views.

## Citation

If you use this code, please cite the PrismAIP paper (reference to be added) and
the ESM-2 backbone (Lin et al., 2023). The adapter follows TLoRA
(Lin et al., 2026): A is initialised from the leading right singular vectors of
`W0 @ C` and frozen, only B is trained.

## License

MIT; see `LICENSE`.

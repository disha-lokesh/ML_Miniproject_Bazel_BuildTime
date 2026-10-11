# Predicting Bazel Build Times with Machine Learning

UE24CS352A Machine Learning — Mini Project

We reproduce, audit and improve *"Predicting Bazel Build Times Using Machine Learning"*
(Nadeem & Raza, Stanford CS229). Given the files changed by a commit, the model predicts the
CPU time of the incremental `bazel build //src:bazel-dev` that follows.

## Headline results (test set = builds 401–500, same split as the paper)

| Model | RMSE (ms) | MAE (ms) | Spearman ρ | AUC (build > 5 s) |
|---|---:|---:|---:|---:|
| Paper's test code (untrained model, predicts 0) | 7,459 | – | – | – |
| Always predict the training mean | 5,807 | 4,352 | 0.00 | 0.50 |
| Paper model reproduced (one-hot prefix × type, SGD) | 5,876 | 4,394 | 0.09 | 0.55 |
| Our features, **paper's commit alignment** | 6,362 | 3,822 | −0.05 | 0.45 |
| **Our features, fixed alignment (gradient boosting, final model)** | **4,954** | **2,744** | **0.82** | **0.85** |

The final model is trained on train + validation after model selection. 5-fold CV over all 499 incremental builds: ours RMSE 4,709 ± 834, ρ 0.78 ± 0.03;
paper features RMSE 5,315 ± 728, ρ 0.00 ± 0.08.
All numbers are produced by `src/train.py` and stored in `results/metrics.json`.

## What we found

1. **No signal in the original model.** It is no better than predicting the mean.
2. **Test-evaluation bug.** `test_model()` in the original code builds a fresh, untrained
   `LinearRegressor`, so every test prediction is 0. That is the flat ~7,450 ms test line in the paper.
3. **Feature-extraction bugs.** Types were counted with substrings (`".c"` matches `.cc`/`.css`,
   `.h` matches `.html`, `.js` matches `.json`) and a commit with no listed type defaulted to `JAVA`.
   `"tools/"` matched every `com/google/devtools/` path.
4. **Off-by-one alignment.** The dataset was built by walking history backwards (`git checkout HEAD~1`)
   with a warm Bazel cache. Build *i* therefore re-does the work caused by the change between commit *i*
   and commit *i−1* — i.e. the files changed **by the previous row's commit** — not commit *i*'s own diff.
   Re-aligning is the single biggest improvement.
5. **CPU time ≈ how much of the action graph re-runs** (Spearman 0.92 with `actionsExecuted` from the
   Build Event Protocol). 149 of 500 builds re-ran just one action (docs/test-only changes).

## Repository layout

```
data/raw/InputData.csv          paper's features (commit id, prefix, type)       [from original repo]
data/raw/CPUTimes.csv           target: CPU ms per build                         [from original repo]
data/raw/commit_sequence.txt    full SHAs of the 500 commits (+ parent)
data/processed/commit_diffs.csv every changed file + lines added/deleted per commit (ours)
data/processed/bep_metrics.csv  CPU ms, wall ms, actions executed per build (ours)
scripts/enrich_from_git.py      rebuilds commit_diffs.csv from a bazelbuild/bazel clone
scripts/extract_bep_metrics.py  rebuilds bep_metrics.csv from the original BEP JSON files
src/features.py                 path -> area/type classification, feature builder, alignment
src/models.py                   from-scratch SGD linear regression, paper design matrix, our models
src/train.py                    runs all experiments, writes metrics.json + figures + model.joblib
src/predict.py                  live demo CLI
results/                        metrics.json, model.joblib, figures/
```

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                                    # ~15 s, regenerates everything in results/
```

### Live demo

```bash
# a core Java change -> medium/expensive rebuild
python src/predict.py --files src/main/java/com/google/devtools/build/lib/exec/SpawnRunner.java

# docs + test change -> near no-op rebuild
python src/predict.py --files site/docs/install.md src/test/shell/bazel/foo_test.sh

# replay real test builds and compare with the measured CPU time
python src/predict.py --row 407     # 11-file refactor incl. a .proto -> EXPENSIVE (actual 27 s)
python src/predict.py --row 415     # test-only change -> CHEAP (actual 0.6 s)

# from a real Bazel commit (needs a clone, see below)
python src/predict.py --repo /tmp/bazel --commit c25a504ec4
```

### Rebuilding the processed data (optional)

```bash
git clone --filter=blob:none --no-checkout https://github.com/bazelbuild/bazel.git /tmp/bazel
python scripts/enrich_from_git.py --repo /tmp/bazel            # ~4 min

git clone https://github.com/umar-nadeem/CS229-Final-Project /tmp/orig
python scripts/extract_bep_metrics.py --bep-dir /tmp/orig/Build_Info
```

## Method in one paragraph

Each commit's changed files are mapped to a source area (`src/main/java`, `src/test`, `site`, …) and a
file type by real extension. Features: per-area and per-type file counts, log(#files), log(lines
added/deleted), and flags (touches `src/main`, only tests/docs, touches BUILD/.bzl). Row *i* gets the
diff of commit *i−1*. The target is modelled as log(CPU ms) because it is heavily right-skewed
(median 1.8 s, max 45 s). We compare ridge regression, random forest and gradient boosting, select on
validation RMSLE, and report test once. Build 1 (cold cache, 45 s) is excluded from training.

## Limitations

- 500 builds from one machine, one target, one month of history; absolute times won't transfer.
- The model under-predicts the very largest rebuilds (> 15 s); it ranks them well but compresses the tail.
- File paths are a proxy for what really matters — the size of the reverse-dependency closure of the
  changed targets (`bazel query 'rdeps(//src:bazel-dev, ...)'`), the natural next feature.

## Credits

Original paper, data collection and BEP files: Muhammad Umar Nadeem and Sarah Raza,
<https://github.com/umar-nadeem/CS229-Final-Project>. Commit history: <https://github.com/bazelbuild/bazel>.

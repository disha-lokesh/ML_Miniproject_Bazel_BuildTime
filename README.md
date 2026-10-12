# Predicting Bazel Build Times with Machine Learning

UE24CS352A Machine Learning, Mini Project

We reproduce, audit and improve *"Predicting Bazel Build Times Using Machine Learning"*
(Nadeem & Raza, Stanford CS229). Given the files changed by a commit, the model predicts the
CPU time of the incremental `bazel build //src:bazel-dev` that follows.

## Headline results (test set = builds 401 to 500, same split as the paper)

| Model | RMSE (ms) | MAE (ms) | Spearman ρ | AUC (build > 5 s) |
|---|---:|---:|---:|---:|
| Paper's test code (untrained model, predicts 0) | 7,459 | n/a | n/a | n/a |
| Always predict the training mean | 5,807 | 4,352 | 0.00 | 0.50 |
| Paper model reproduced (one-hot prefix × type, SGD) | 5,876 | 4,394 | 0.09 | 0.55 |
| Our diff features, **paper's commit alignment** | 6,362 | 3,822 | −0.05 | 0.45 |
| Our diff features, fixed alignment | 4,954 | 2,744 | 0.82 | 0.85 |
| **+ size/dependency features (final model, chosen on validation)** | **5,143** | **2,811** | **0.81** | **0.83** |

The feature set and the model are chosen together on validation RMSLE only; the size features won
(0.681 vs 0.708) and the final model is retrained on train + validation and scored once on test.
On these 100 test builds it is slightly behind the diff-only model, a gap well inside the
fold-to-fold spread (about ±700 to 850 ms RMSE). Over all 499 incremental builds it is clearly ahead:

| 5-fold CV, 499 builds | RMSE (ms) | Spearman ρ |
|---|---:|---:|
| Paper features (SGD) | 5,315 ± 728 | 0.00 ± 0.08 |
| Diff features, fixed alignment | 4,709 ± 834 | 0.78 ± 0.03 |
| **+ size/dependency features (final)** | **4,210 ± 701** | **0.80 ± 0.03** |

In a paired comparison on identical folds (20 repetitions × 5 folds) adding the size features lowers
RMSLE by 0.040 on average, in 95% of the 100 folds and in all 20 repetitions.
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
   and commit *i−1*, i.e. the files changed **by the previous row's commit**, not commit *i*'s own diff.
   Re-aligning is the single biggest improvement.
5. **CPU time ≈ how much of the action graph re-runs** (Spearman 0.92 with `actionsExecuted` from the
   Build Event Protocol). 149 of 500 builds re-ran just one action (docs/test-only changes).
6. **How big the changed files are helps.** Bytes and the number of imports / `#include`s of the
   changed files add signal on top of *which* files changed (paired CV above). We use the files'
   state *before* the commit, because that is the version the backwards walk actually compiles.
7. **Cheap file-based cost estimates are weaker than simple features.** Two hand-built cost proxies
   from a companion project that mined all 47k Bazel commits (one with fixed language/directory
   weights, one fitted to measured `javac`/`g++` compile times) correlate with the real CPU time at
   ρ 0.66 and 0.53, below the single feature *files in src/main/java* (0.72). Adding them as
   features does not help on validation. Measured compile times of individual files do not transfer
   to what a Bazel build re-runs.

## Repository layout

```
data/raw/InputData.csv            paper's features (commit id, prefix, type)       [from original repo]
data/raw/CPUTimes.csv             target: CPU ms per build                         [from original repo]
data/raw/commit_sequence.txt      full SHAs of the 500 commits (+ parent)
data/processed/commit_diffs.csv   every changed file + lines added/deleted per commit
data/processed/bep_metrics.csv    CPU ms, wall ms, actions executed per build
data/processed/size_features.csv  bytes and imports/#includes of changed files, before and after state
data/processed/proxy_costs.csv    two file-based cost proxies per commit (companion project)
scripts/enrich_from_git.py        rebuilds commit_diffs.csv from a bazelbuild/bazel clone
scripts/extract_bep_metrics.py    rebuilds bep_metrics.csv from the original BEP JSON files
scripts/size_features.py          rebuilds size_features.csv from a bazelbuild/bazel clone
scripts/proxy_costs.py            rebuilds proxy_costs.csv from the companion project's data
src/features.py                   path -> area/type classification, feature builder, alignment
src/deps.py                       counts imports (Java) and #includes (C/C++) in a file
src/models.py                     from-scratch SGD linear regression, paper design matrix, our models
src/train.py                      runs all experiments, writes metrics.json + figures + model.joblib
src/predict.py                    live demo CLI
results/                          metrics.json, model.joblib, figures/
docs/                             two-page write-up and slide deck, with their generators
```

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                                    # about 4-5 min, regenerates everything in results/
```

To rebuild the write-up and slides from the results:

```bash
pip install reportlab pillow python-pptx
python docs/make_writeup.py      # docs/writeup.pdf (2 pages)
python docs/make_slides.py       # docs/slides.pptx
```

Most of the training run time is the paired cross-validation (400 model fits). The processed data is
committed, so no Bazel clone is needed to train or to run the demo from a dataset row or a file list.

### Live demo

```bash
# a core Java change -> medium/expensive rebuild
python src/predict.py --files src/main/java/com/google/devtools/build/lib/exec/SpawnRunner.java

# docs + test change -> near no-op rebuild
python src/predict.py --files site/docs/install.md src/test/shell/bazel/foo_test.sh

# replay real test builds and compare with the measured CPU time
python src/predict.py --row 407     # 11-file refactor incl. a .proto -> EXPENSIVE (actual 27 s)
python src/predict.py --row 415     # test-only change -> CHEAP (actual 0.6 s)

# from a real Bazel commit (needs a clone, see below); uses the size features too
python src/predict.py --repo /tmp/bazel --commit c25a504ec4
```

With `--files` the file sizes are unknown, so the demo says so and uses a model trained on the diff
features only. `--row` and `--commit` use the final model.

### Rebuilding the processed data (optional)

```bash
git clone --no-checkout https://github.com/bazelbuild/bazel.git /tmp/bazel   # full clone, ~1 GB
python scripts/enrich_from_git.py --repo /tmp/bazel            # ~4 min
python scripts/size_features.py --repo /tmp/bazel              # ~2 min

git clone https://github.com/umar-nadeem/CS229-Final-Project /tmp/orig
python scripts/extract_bep_metrics.py --bep-dir /tmp/orig/Build_Info

# proxies: needs the companion project's data/target_track_a.csv and target_track_b.csv
python scripts/proxy_costs.py --miniproject ../miniproject
```

`enrich_from_git.py` also works with a blobless clone (`--filter=blob:none`); `size_features.py`
reads file contents, so a blobless clone works but fetches each blob over the network.

## Method in one paragraph

Each commit's changed files are mapped to a source area (`src/main/java`, `src/test`, `site`, …) and a
file type by real extension. Diff features: per-area and per-type file counts, log(#files), log(lines
added/deleted), and flags (touches `src/main`, only tests/docs, touches BUILD/.bzl). Size features:
log of the total and largest size in bytes and of the total and largest number of imports/`#include`s
of the changed files, overall and for Java and C/C++ separately, in the state before the commit.
Row *i* gets the features of commit *i−1*. The target is modelled as log(CPU ms) because it is heavily
right-skewed (median 1.8 s, max 45 s). We compare ridge regression, random forest and gradient boosting
on each feature set, select feature set and model together on validation RMSLE, and report test once.
Build 1 (cold cache, 45 s) is excluded from training.

## Limitations

- 500 builds from one machine, one target, one month of history; absolute times won't transfer.
- The test set has only 100 builds, so single-split differences of a few percent are noise; we rely
  on repeated, paired cross-validation for the size-feature comparison.
- The model under-predicts the very largest rebuilds (> 15 s); it ranks them well but compresses the tail.
- File paths and sizes are still a proxy for what really matters, the size of the reverse-dependency
  closure of the changed targets (`bazel query 'rdeps(//src:bazel-dev, ...)'`), the natural next feature.

## Credits

Original paper, data collection and BEP files: Muhammad Umar Nadeem and Sarah Raza,
<https://github.com/umar-nadeem/CS229-Final-Project>. Commit history: <https://github.com/bazelbuild/bazel>.

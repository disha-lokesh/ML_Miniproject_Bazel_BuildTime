# Validation report

What was checked before submission, how, and the result. Re-run the automated part with
`python -m unittest discover tests -v` (add `BAZEL_REPO=/path/to/bazel` for the git checks).

## 1. Scope: mini-project guidelines (UE24CS352A) and problem 60

| Requirement | Status | Evidence |
|---|---|---|
| Problem 60, "Predicting Bazel Build Times Using Machine Learning" | Met | same task, data and split as the CS229 paper |
| Code in a **private** GitHub repository, shared with faculty/TAs | **Not met yet** | the repository page loads without login (public); switch to private and add collaborators |
| README with setup and run instructions | Met | `README.md`: setup, training, demo, data rebuild, docs rebuild |
| PDF write-up: problem, dataset, approach, implementation, conclusions | Met | `docs/writeup.pdf`, sections 1 to 6; 2 pages (test `test_writeup_is_two_pages`) |
| Slide deck for the review | Met | 14-slide presentation, every number taken from `results/metrics.json`; exported to `docs/slides.pdf` |
| Live demonstration of working code | Met | `src/predict.py`, 4 modes; tests `Demo.*` |
| Individual contribution visible | Met | git authorship per commit; "Who did what" slide |

## 2. Data provenance

| Check | Result |
|---|---|
| `data/raw/InputData.csv`, `CPUTimes.csv` vs the original repository | byte-identical |
| `bep_metrics.csv` rebuilt from the 500 original BEP JSON files | byte-identical |
| `commit_diffs.csv`, `commit_sequence.txt` rebuilt from the Bazel history | identical content |
| `size_features.csv` rebuilt from the Bazel history | identical content |
| `size_features.csv` "after" state vs features mined independently in the companion project | identical for all 500 commits |
| `proxy_costs.csv` vs the companion project's targets | identical |
| Original extractor (`extractBuildInfo.py`, unchanged) run on the recovered file lists | reproduces `InputData.csv` for **500/500** prefixes and **500/500** types |

The last row proves both that the commit mapping and diffs are right, and that the paper's features
were computed from each commit's own diff (the premise of the alignment fix).

## 3. Audit claims about the original code, verified against its source

| Claim | Verified | How |
|---|---|---|
| `test_model()` evaluates an untrained model | yes | it creates a new `LinearRegressor` (optimizer `Ftrl`) and only calls `predict`; RMSE of all-zero predictions = 7,459 ms, the paper's flat test line |
| `.c` matches `.cc`/`.css`, `.h` matches `.html`, `.js` matches `.json` | yes | `getType` uses `str.count` on the raw path list |
| Commits with no counted type default to `JAVA` | yes | `max()` over all-zero counts returns the first key; **68** commits |
| `'tools/'` matches `com/google/devtools/` paths | yes | **144 of 171** `tools/` labels have no file under `tools/` |
| Off-by-one between features and builds | yes | the script runs `git checkout HEAD~1`, takes `git diff HEAD~1..HEAD` (the new commit's own diff), then builds from the previously built child; the build re-does the child's changes. With the shift, Spearman of *touches src/main* goes from 0.03 to 0.70 |

Found during validation, not previously reported:

- `getType` counts the literal string `"*.bzl"`, which never occurs: `.bzl` files (22 commits) are never counted as Starlark.
- 22 commits are labelled `C/C++` without containing any C/C++ file.
- The original split uses `head(300)`, `loc[301:400]`, `tail(100)`: row 300 is unused and row 400 is in both validation and test. This repository uses rows 0-299 / 300-399 / 400-499.

Known deviations of our paper reproduction (they do not change the conclusion that the model has no signal):
our SGD runs 10 periods x 10 epochs (1,900 steps, the paper ran 500) and builds the feature cross as an
explicit 55-way one-hot instead of TensorFlow's 1,000-bucket hash.

## 4. Our modelling work

| Check | Result |
|---|---|
| No leakage: model inputs known before building (no CPU, wall time, actions, IDs) | test `NoLeakage` |
| Feature set and model chosen on validation only | test `test_selection_uses_validation_only` |
| Final and diff-only test scores reproduce by retraining | test `test_final_model_test_score_reproduces` (5,143 and 4,954 ms) |
| Saved model gives the reported test score | test `test_saved_model_matches_retraining` |
| Alignment is exactly a one-row shift for every feature | test `test_previous_alignment_is_a_one_row_shift` |
| Path classification has none of the original substring bugs | tests `FileClassification.*` |
| Size features recomputed from git for sample rows, via the demo's code path | test `AgainstGit` |
| Every headline number in the README equals `results/metrics.json` | test `test_headline_numbers` |
| Full pipeline from a fresh clone | `metrics.json`, figures and model byte-identical to the committed ones |

## 5. Open items

- Make the repository private and share it with faculty and TAs (guideline requirement).
- The submission deadline (10 Oct 2026, 23:59) has passed; confirm a late submission with the faculty.
- The size-feature gain is shown by paired cross-validation; on the 100-build test split the final model
  is slightly behind the diff-only model (5,143 vs 4,954 ms RMSE), within fold-to-fold noise.
- Fill in names and SRNs on the write-up, the title slide and the contributions slide (left blank on purpose).

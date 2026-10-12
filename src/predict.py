"""
Live demo: predict the CPU time of the incremental Bazel build that follows a change.

Examples
  # from a list of changed files (no git needed)
  python src/predict.py --files src/main/java/com/google/devtools/build/lib/exec/SpawnRunner.java
  python src/predict.py --files site/docs/install.md src/test/shell/bazel/foo_test.sh

  # from a real commit in a local bazelbuild/bazel clone
  python src/predict.py --repo /tmp/bazel --commit c25a504ec4

  # replay a build from the dataset and compare with the real value
  python src/predict.py --row 37
"""
import argparse
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import SIZE_STATS, area_of, commit_features, load_dataset, type_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def bucket(ms):
    if ms < 1500:
        return "CHEAP   (near no-op rebuild)"
    if ms < 5000:
        return "MEDIUM"
    return "EXPENSIVE (large part of the graph rebuilds)"


def files_from_commit(repo, commit):
    out = subprocess.run(["git", "-C", repo, "diff", "--numstat", "--no-renames", f"{commit}^", commit],
                         capture_output=True, text=True, check=True).stdout
    files, add, dele = [], 0, 0
    for line in out.splitlines():
        a, d, p = line.split("\t", 2)
        files.append(p)
        add += int(a) if a != "-" else 0
        dele += int(d) if d != "-" else 0
    return files, add, dele


def size_features_from_commit(repo, commit):
    """log1p size/dependency features of the files a commit changes (both states)."""
    from size_features import aggregate, blob_info, changed_blobs  # noqa: E402
    from pathlib import PurePosixPath
    files = changed_blobs(repo, f"{commit}^", commit)
    blobs = {}
    for path, old, new in files:
        blobs[old] = blobs[new] = PurePosixPath(path).suffix.lower()
    info = blob_info(repo, blobs)
    out = {}
    for which in ("before", "after"):
        agg = aggregate(files, info, which)
        out.update({f"log_{which}_{c}": float(np.log1p(agg[c])) for c in SIZE_STATS})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", help="changed file paths")
    ap.add_argument("--lines-added", type=int, default=20)
    ap.add_argument("--lines-deleted", type=int, default=5)
    ap.add_argument("--repo", help="path to a bazelbuild/bazel clone")
    ap.add_argument("--commit", help="commit whose changes to predict for")
    ap.add_argument("--row", type=int, help="dataset row (0-499) to replay")
    args = ap.parse_args()

    bundle = joblib.load(os.path.join(ROOT, "results", "model.joblib"))
    model, cols = bundle["model"], bundle["columns"]

    actual = None
    if args.row is not None:
        df = load_dataset(ROOT, align="previous")
        r = df.iloc[args.row]
        x = r[cols].values.astype(float)[None, :]
        actual = r["cpu_ms"]
        diffs = pd.read_csv(os.path.join(ROOT, "data/processed/commit_diffs.csv"), keep_default_na=False)
        files = [f for f in diffs.iloc[args.row - 1]["files"].split("|") if f] if args.row > 0 else []
        split = "train" if args.row < 300 else ("validation" if args.row < 400 else "test")
        print(f"Build #{args.row + 1} ({split} row) -- rebuild triggered by commit {diffs.iloc[args.row - 1]['sha'][:10]}")
    else:
        if args.commit:
            files, add, dele = files_from_commit(args.repo, args.commit)
        elif args.files:
            files, add, dele = args.files, args.lines_added, args.lines_deleted
        else:
            ap.error("give --files, --commit or --row")
        feats = commit_features(files, add, dele)
        if args.commit:
            feats.update(size_features_from_commit(args.repo, args.commit))
        if not all(c in feats for c in cols):
            # file sizes unknown without a repository: use the model trained on diff features only
            model, cols = bundle["fallback_model"], bundle["fallback_columns"]
            print("(no repository given, so file sizes are unknown: using the diff-features-only model)")
        x = np.array([[feats[c] for c in cols]], dtype=float)

    pred = float(np.exp(model.predict(x)[0]))
    print(f"\nChanged files ({len(files)}):")
    for f in files[:12]:
        print(f"  [{area_of(f):18s} | {type_of(f):8s}] {f}")
    if len(files) > 12:
        print(f"  ... and {len(files) - 12} more")
    print(f"\nPredicted CPU time : {pred:8.0f} ms  -> {bucket(pred)}")
    if actual is not None:
        print(f"Actual CPU time    : {actual:8.0f} ms  -> {bucket(actual)}")


if __name__ == "__main__":
    main()

"""
Size and dependency features for each of the 500 commits.

For every file a commit changes we read its size in bytes and the number of
imports (Java) or #includes (C/C++) it declares, in two states:

  before : the file in the commit's parent
  after  : the file in the commit itself

Because the original dataset walks history backwards, build i compiles the
files changed by commit i-1 in their BEFORE state (see src/features.py), so
"before" is the physically correct choice; "after" is kept for comparison.
A file that does not exist in a state (added or deleted) counts 0 there.

Usage (a full clone is fastest; a blobless clone works but fetches each blob):
    git clone --no-checkout https://github.com/bazelbuild/bazel.git /tmp/bazel
    python scripts/size_features.py --repo /tmp/bazel

Output: data/processed/size_features.csv (one row per commit, same order as InputData.csv)
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deps import CODE_EXT, CPP_EXT, JAVA_EXT, count_deps  # noqa: E402

ZERO = "0" * 40
STATS = ["sum_bytes", "max_bytes", "sum_deps", "max_deps",
         "java_bytes", "java_deps", "cpp_bytes", "cpp_deps"]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, "-c", "core.quotepath=off", *args],
                          capture_output=True, text=True, check=True).stdout


def changed_blobs(repo, old, new):
    """[(path, old_blob, new_blob)] for the diff old -> new."""
    out = []
    for line in git(repo, "diff", "--raw", "--no-renames", "--no-abbrev", old, new).splitlines():
        meta, path = line.split("\t", 1)
        parts = meta.split()
        out.append((path, parts[2], parts[3]))
    return out


def blob_info(repo, blobs):
    """{blob: (size, n_deps)}; blobs maps blob id -> file extension."""
    ids = sorted(b for b in blobs if b != ZERO)
    check = subprocess.run(["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objectsize)"],
                           input="\n".join(ids) + "\n", capture_output=True, text=True, check=True)
    info = {}
    for line in check.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            info[parts[0]] = (int(parts[1]), 0)
    for b in ids:
        if blobs[b] in CODE_EXT and b in info:
            content = subprocess.run(["git", "-C", repo, "cat-file", "blob", b],
                                     capture_output=True, check=True).stdout
            info[b] = (info[b][0], count_deps(content, blobs[b]))
    return info


def aggregate(files, info, which):
    s = dict.fromkeys(STATS, 0)
    for path, old, new in files:
        blob = old if which == "before" else new
        size, deps = info.get(blob, (0, 0))
        ext = PurePosixPath(path).suffix.lower()
        s["sum_bytes"] += size
        s["sum_deps"] += deps
        s["max_bytes"] = max(s["max_bytes"], size)
        s["max_deps"] = max(s["max_deps"], deps)
        if ext in JAVA_EXT:
            s["java_bytes"] += size
            s["java_deps"] += deps
        if ext in CPP_EXT:
            s["cpp_bytes"] += size
            s["cpp_deps"] += deps
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to bazelbuild/bazel clone")
    args = ap.parse_args()

    seq = (ROOT / "data/raw/commit_sequence.txt").read_text().split()
    per_commit = []
    blobs = {}
    for i, sha in enumerate(seq[:-1]):
        files = changed_blobs(args.repo, seq[i + 1], sha)   # changes made BY commit i
        per_commit.append(files)
        for path, old, new in files:
            ext = PurePosixPath(path).suffix.lower()
            blobs[old] = blobs[new] = ext
    info = blob_info(args.repo, blobs)

    out = ROOT / "data/processed/size_features.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "sha"] + [f"{which}_{s}" for which in ("before", "after") for s in STATS])
        for i, files in enumerate(per_commit):
            b, a = aggregate(files, info, "before"), aggregate(files, info, "after")
            w.writerow([i, seq[i]] + [b[s] for s in STATS] + [a[s] for s in STATS])
    print("wrote", out, f"({len(per_commit)} commits, {len(blobs)} blobs)")


if __name__ == "__main__":
    main()

"""
Recover the full git diff for each of the 500 Bazel commits used in the
original CS229 dataset, so we can build richer features than
"most common prefix" and "most common file type".

Why this is possible: the original InputData.csv stores each commit's short
hash (minus its first character). Walking `git log --first-parent` back from
the June-2021 HEAD, all 500 hashes line up one-to-one, in order.

Usage (needs a clone of bazelbuild/bazel; a blobless clone is enough):
    git clone --filter=blob:none --no-checkout https://github.com/bazelbuild/bazel.git /tmp/bazel
    python scripts/enrich_from_git.py --repo /tmp/bazel

Output: data/processed/commit_diffs.csv  (one row per commit, same order as InputData.csv)
"""
import argparse
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, check=True).stdout


def find_sequence(repo, short_ids):
    """Return the 501 full SHAs (500 commits + parent of the last one)."""
    shas = git(repo, "log", "--first-parent", "--format=%H",
               "--until=2021-06-03").split()
    first = short_ids[0]
    start = next(i for i, h in enumerate(shas) if h[1:1 + len(first)] == first)
    seq = shas[start:start + len(short_ids) + 1]
    for sid, sha in zip(short_ids, seq):
        assert sha[1:1 + len(sid)] == sid, f"history mismatch at {sid}"
    return seq


def diff_stats(repo, old, new):
    """Files changed between two commits, with added/deleted line counts."""
    rows = []
    for line in git(repo, "diff", "--numstat", "--no-renames", old, new).splitlines():
        added, deleted, path = line.split("\t", 2)
        # binary files show "-"
        rows.append((path, int(added) if added != "-" else 0,
                     int(deleted) if deleted != "-" else 0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to bazelbuild/bazel clone")
    args = ap.parse_args()

    with open(ROOT / "data/raw/InputData.csv") as f:
        short_ids = [r["ID"] for r in csv.DictReader(f)]

    seq = find_sequence(args.repo, short_ids)
    (ROOT / "data/raw/commit_sequence.txt").write_text("\n".join(seq) + "\n")

    out = ROOT / "data/processed/commit_diffs.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "sha", "n_files", "lines_added", "lines_deleted", "files"])
        for i, sha in enumerate(seq[:-1]):
            stats = diff_stats(args.repo, seq[i + 1], sha)  # changes made BY commit i
            w.writerow([i, sha, len(stats), sum(s[1] for s in stats),
                        sum(s[2] for s in stats), "|".join(s[0] for s in stats)])
            if i % 50 == 0:
                print(f"{i}/500")
    print("wrote", out)


if __name__ == "__main__":
    main()

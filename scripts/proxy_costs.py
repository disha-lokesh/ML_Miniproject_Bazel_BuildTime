"""
Export two file-based build-cost proxies for the 500 commits, to check how well
cheap estimates track the real CPU time.

The proxies come from the companion project that mined the full bazelbuild/bazel
history (47k commits), where no real build time exists:
  cost_a : fixed formula, language weight x directory weight x (1 + ln(1 + lines changed))
  time_b : seconds to recompile the changed files, fitted on measured javac / g++ runs

Usage:
    python scripts/proxy_costs.py --miniproject ../miniproject

Needs data/target_track_a.csv and data/target_track_b.csv from that project
(regenerate them there with run_all.py).
Output: data/processed/proxy_costs.csv (one row per commit, same order as InputData.csv)
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miniproject", required=True, help="path to the companion project")
    args = ap.parse_args()
    mp = Path(args.miniproject) / "data"

    seq = (ROOT / "data/raw/commit_sequence.txt").read_text().split()[:500]
    a = pd.read_csv(mp / "target_track_a.csv").set_index("sha")["cost_a"]
    b = pd.read_csv(mp / "target_track_b.csv").set_index("sha")["time_b"]
    missing = [s for s in seq if s not in a.index or s not in b.index]
    if missing:
        raise SystemExit(f"{len(missing)} commits missing from the companion data, e.g. {missing[0]}")

    out = pd.DataFrame({"row": range(500), "sha": seq,
                        "cost_a": a.reindex(seq).values, "time_b": b.reindex(seq).values})
    out.to_csv(ROOT / "data/processed/proxy_costs.csv", index=False)
    print("wrote data/processed/proxy_costs.csv")


if __name__ == "__main__":
    main()

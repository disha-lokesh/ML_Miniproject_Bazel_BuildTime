"""
Pull CPU time, wall time and number of executed actions out of the Build
Event Protocol (BEP) JSON files saved by the original authors.

The 500 BEP files (~55 MB) live in the original repo, not this one:
    git clone https://github.com/umar-nadeem/CS229-Final-Project /tmp/orig
    python scripts/extract_bep_metrics.py --bep-dir /tmp/orig/Build_Info

Output: data/processed/bep_metrics.csv

Note: actions_executed and wall time are only known AFTER a build finishes,
so they are used for analysis only, never as model inputs (that would leak
the answer into the features).
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_metrics(path):
    # BEP JSON is newline-delimited: one event per line
    for line in open(path):
        event = json.loads(line)
        if "buildMetrics" in event:
            m = event["buildMetrics"]
            return (int(m["timingMetrics"]["cpuTimeInMs"]),
                    int(m["timingMetrics"].get("wallTimeInMs", 0)),
                    int(m.get("actionSummary", {}).get("actionsExecuted", 0)))
    raise ValueError(f"no buildMetrics in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bep-dir", required=True)
    args = ap.parse_args()
    out = ROOT / "data/processed/bep_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["build", "cpu_ms", "wall_ms", "actions_executed"])
        for i in range(1, 501):
            w.writerow([i, *read_metrics(Path(args.bep_dir) / f"{i}_commit.json")])
    print("wrote", out)


if __name__ == "__main__":
    main()

"""
Feature engineering for Bazel build-time prediction.

Two feature sets:

1. ORIGINAL (paper): the single most common path prefix and the single most
   common file type in a commit, one-hot encoded, plus their feature cross.

2. ENRICHED (ours): computed from the full list of changed files of a commit:
   - how many files fall in each source area (src/main/java, src/test, site, ...)
   - how many files of each type (Java, C++, Starlark, proto, ...)
   - total files changed, lines added, lines deleted (log-scaled)
   - flags such as "touches a BUILD/.bzl file" or "only touches docs/tests"

Alignment fix
-------------
The authors walked history BACKWARDS: build i checks out commit i right after
building commit i-1 (its child). Bazel is incremental, so the work done in
build i is caused by the files that differ between commit i and commit i-1,
which are the files changed BY commit i-1, not by commit i.
`align="previous"` therefore gives row i the diff of row i-1.
"""
import os
import numpy as np
import pandas as pd

# Source areas, most specific first (first match wins)
AREAS = [
    ("src/main/java", "src/main/java/"),
    ("src/main/cpp", "src/main/cpp/"),
    ("src/main/native", "src/main/native/"),
    ("src/main/protobuf", "src/main/protobuf/"),
    ("src/main/starlark", "src/main/starlark/"),
    ("src/main/other", "src/main/"),
    ("src/test", "src/test/"),
    ("src/tools", "src/tools/"),
    ("src/java_tools", "src/java_tools/"),
    ("src/conditions", "src/conditions/"),
    ("src/other", "src/"),
    ("site", "site/"),
    ("tools", "tools/"),
    ("third_party", "third_party/"),
    ("examples", "examples/"),
    ("scripts", "scripts/"),
    ("ci", ".bazelci/"),
]

TYPES = {
    "java": {".java"},
    "cpp": {".c", ".cc", ".cpp", ".h", ".hpp"},
    "proto": {".proto"},
    "python": {".py"},
    "shell": {".sh"},
    "docs": {".md", ".html", ".css", ".js", ".txt", ".yaml", ".yml", ".json"},
}
STARLARK_NAMES = {"BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel"}
STARLARK_EXT = {".bzl", ".bazel"}


def area_of(path):
    for name, prefix in AREAS:
        if path.startswith(prefix):
            return name
    return "root_other"


def type_of(path):
    """Classify by real file extension (the original code used substring
    counts, so '.c' also matched '.cc', '.css', '.cpp', and '.js' matched '.json')."""
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    if base in STARLARK_NAMES or ext in STARLARK_EXT:
        return "starlark"
    for t, exts in TYPES.items():
        if ext in exts:
            return t
    return "other"


AREA_NAMES = [a for a, _ in AREAS] + ["root_other"]
TYPE_NAMES = list(TYPES) + ["starlark", "other"]


def commit_features(files, lines_added, lines_deleted):
    """Feature dict for one commit, given its list of changed file paths."""
    feats = {f"area_{a}": 0 for a in AREA_NAMES}
    feats.update({f"type_{t}": 0 for t in TYPE_NAMES})
    for p in files:
        feats[f"area_{area_of(p)}"] += 1
        feats[f"type_{type_of(p)}"] += 1
    n = len(files)
    feats["log_n_files"] = np.log1p(n)
    feats["log_lines_added"] = np.log1p(lines_added)
    feats["log_lines_deleted"] = np.log1p(lines_deleted)
    feats["touches_build_files"] = int(feats["type_starlark"] > 0)
    # changes that cannot affect the //src:bazel-dev binary
    non_binary = feats["area_site"] + feats["area_src/test"] + feats["area_examples"] + feats["area_ci"]
    feats["only_tests_or_docs"] = int(n > 0 and non_binary == n)
    feats["touches_src_main"] = int(sum(feats[f"area_{a}"] for a in AREA_NAMES
                                        if a.startswith("src/main")) > 0)
    return feats


def load_dataset(root, align="previous"):
    """
    Returns a DataFrame with one row per build (500), containing:
      prefix, type          -- original paper features
      enriched features     -- from commit_diffs.csv
      cpu_ms                -- target
      cold_build            -- True for build 1 (empty cache)
    align: "previous" (correct for the backwards walk) or "same" (as in paper)
    """
    raw = pd.read_csv(os.path.join(root, "data/raw/InputData.csv"))
    y = pd.read_csv(os.path.join(root, "data/raw/CPUTimes.csv"))["CPUTime"].values
    diffs = pd.read_csv(os.path.join(root, "data/processed/commit_diffs.csv"),
                        keep_default_na=False)

    feats = []
    for _, r in diffs.iterrows():
        files = [f for f in r["files"].split("|") if f]
        feats.append(commit_features(files, r["lines_added"], r["lines_deleted"]))
    feats = pd.DataFrame(feats)

    # size/dependency features and cost proxies, per commit (scripts/size_features.py,
    # scripts/proxy_costs.py); log-scaled and aligned exactly like the diff features
    size_path = os.path.join(root, "data/processed/size_features.csv")
    if os.path.exists(size_path):
        size = pd.read_csv(size_path)
        for c in SIZE_STATS:
            for which in ("before", "after"):
                feats[f"log_{which}_{c}"] = np.log1p(size[f"{which}_{c}"].values)
    proxy_path = os.path.join(root, "data/processed/proxy_costs.csv")
    if os.path.exists(proxy_path):
        proxy = pd.read_csv(proxy_path)
        feats["log_proxy_cost_a"] = np.log(proxy["cost_a"].values)
        feats["log_proxy_time_b"] = np.log(proxy["time_b"].values)

    if align == "previous":
        # row i <- diff of commit i-1; row 0 (cold build) has no previous diff
        feats = feats.shift(1).fillna(0)

    df = pd.concat([raw[["ID", "prefix", "type"]], feats], axis=1)
    df["cpu_ms"] = y
    df["cold_build"] = False
    df.loc[0, "cold_build"] = True
    return df


ENRICHED_COLS = ([f"area_{a}" for a in AREA_NAMES] + [f"type_{t}" for t in TYPE_NAMES]
                 + ["log_n_files", "log_lines_added", "log_lines_deleted",
                    "touches_build_files", "only_tests_or_docs", "touches_src_main"])

# bytes and imports/#includes of the changed files (scripts/size_features.py).
# "before" is the state build i actually compiles; "after" is kept for comparison.
SIZE_STATS = ["sum_bytes", "max_bytes", "sum_deps", "max_deps",
              "java_bytes", "java_deps", "cpp_bytes", "cpp_deps"]
SIZE_COLS = {which: [f"log_{which}_{c}" for c in SIZE_STATS] for which in ("before", "after")}
PROXY_COLS = ["log_proxy_cost_a", "log_proxy_time_b"]

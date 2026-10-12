"""
Automated checks for the whole project.

    python -m unittest discover tests -v

Fast checks (seconds) run from the committed data only. Checks that need external clones are
skipped unless their path is given:
    BAZEL_REPO=/path/to/bazel        bazelbuild/bazel clone (full or blobless)
"""
import json
import os
import re
import subprocess
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from features import (ENRICHED_COLS, PROXY_COLS, SIZE_COLS, area_of, commit_features,  # noqa: E402
                      load_dataset, type_of)
from models import make_model  # noqa: E402



def read(path, mode="r"):
    with open(os.path.join(ROOT, path), mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        return f.read()


METRICS = json.loads(read("results/metrics.json"))
TRAIN_FINAL, TEST = np.arange(1, 400), np.arange(400, 500)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


class FileClassification(unittest.TestCase):
    """The substring bugs of the original extractor must not come back."""

    def test_types_use_the_real_extension(self):
        cases = {
            "src/main/cpp/blaze.cc": "cpp", "src/main/cpp/util.h": "cpp",
            "site/css/main.css": "docs", "site/index.html": "docs", "site/data.json": "docs",
            "tools/build_defs/repo/http.bzl": "starlark", "src/BUILD": "starlark",
            "src/main/java/A.java": "java", "scripts/release.py": "python",
            "src/main/protobuf/spawn.proto": "proto", "scripts/ci.sh": "shell",
        }
        for path, expected in cases.items():
            self.assertEqual(type_of(path), expected, path)

    def test_devtools_paths_are_not_tools(self):
        self.assertEqual(area_of("src/main/java/com/google/devtools/build/lib/A.java"), "src/main/java")
        self.assertEqual(area_of("tools/jdk/BUILD"), "tools")

    def test_commit_features_count_files(self):
        f = commit_features(["src/main/java/A.java", "src/test/B.java", "site/x.md"], 10, 2)
        self.assertEqual(f["type_java"], 2)
        self.assertEqual(f["area_src/main/java"], 1)
        self.assertEqual(f["touches_src_main"], 1)
        self.assertEqual(f["only_tests_or_docs"], 0)
        g = commit_features(["src/test/B.java", "site/x.md"], 1, 1)
        self.assertEqual(g["only_tests_or_docs"], 1)


class DataIntegrity(unittest.TestCase):
    def test_500_builds_and_targets_match_bep(self):
        cpu = pd.read_csv(os.path.join(ROOT, "data/raw/CPUTimes.csv"))["CPUTime"].values
        bep = pd.read_csv(os.path.join(ROOT, "data/processed/bep_metrics.csv"))
        self.assertEqual(len(cpu), 500)
        np.testing.assert_array_equal(cpu, bep["cpu_ms"].values)

    def test_commit_ids_match_the_sequence(self):
        ids = pd.read_csv(os.path.join(ROOT, "data/raw/InputData.csv"))["ID"].astype(str)
        seq = read("data/raw/commit_sequence.txt").split()
        self.assertEqual(len(seq), 501)
        for i, short in enumerate(ids):
            # the original stored the short hash without its first character
            self.assertEqual(seq[i][1:1 + len(short)], short, f"row {i}")

    def test_processed_files_are_row_aligned(self):
        seq = read("data/raw/commit_sequence.txt").split()[:500]
        for name in ("commit_diffs.csv", "size_features.csv", "proxy_costs.csv"):
            df = pd.read_csv(os.path.join(ROOT, "data/processed", name), keep_default_na=False)
            self.assertEqual(list(df["sha"]), seq, name)

    def test_size_features_are_consistent(self):
        s = pd.read_csv(os.path.join(ROOT, "data/processed/size_features.csv"))
        for which in ("before", "after"):
            self.assertTrue((s[f"{which}_max_bytes"] <= s[f"{which}_sum_bytes"]).all())
            self.assertTrue((s[f"{which}_java_bytes"] + s[f"{which}_cpp_bytes"] <= s[f"{which}_sum_bytes"]).all())
            self.assertTrue((s[f"{which}_max_deps"] <= s[f"{which}_sum_deps"]).all())
            self.assertTrue((s.filter(like=which) >= 0).all().all())


class Alignment(unittest.TestCase):
    def test_previous_alignment_is_a_one_row_shift(self):
        same, prev = load_dataset(ROOT, "same"), load_dataset(ROOT, "previous")
        cols = ENRICHED_COLS + SIZE_COLS["before"] + SIZE_COLS["after"] + PROXY_COLS
        np.testing.assert_allclose(prev[cols].values[1:], same[cols].values[:-1])
        self.assertTrue((prev[cols].iloc[0] == 0).all())

    def test_alignment_is_what_creates_the_signal(self):
        e3 = METRICS["E3_feature_target_spearman"]["touches_src_main"]
        self.assertLess(abs(e3["paper_alignment"]), 0.1)
        self.assertGreater(e3["fixed_alignment"], 0.6)


class NoLeakage(unittest.TestCase):
    def test_model_inputs_are_known_before_building(self):
        bundle = joblib.load(os.path.join(ROOT, "results", "model.joblib"))
        allowed = set(ENRICHED_COLS) | set(SIZE_COLS["before"]) | set(SIZE_COLS["after"]) | set(PROXY_COLS)
        for cols in (bundle["columns"], bundle["fallback_columns"]):
            self.assertTrue(set(cols) <= allowed)
            for c in cols:
                self.assertNotRegex(c, "cpu|wall|action|ID")


class EvaluationProtocol(unittest.TestCase):
    def test_selection_uses_validation_only(self):
        e5 = METRICS["E5_size_features"]
        chosen = f"{METRICS['selected_model']}|{METRICS['selected_features']}"
        candidates = [k for k in e5 if k.split("|")[1] in ("diff", "diff+size_before")]
        self.assertEqual(chosen, min(candidates, key=lambda k: e5[k]["val"]["rmsle"]))

    def test_final_model_test_score_reproduces(self):
        prev = load_dataset(ROOT, "previous")
        y = prev["cpu_ms"].values.astype(float)
        bundle = joblib.load(os.path.join(ROOT, "results", "model.joblib"))
        for cols, key in ((bundle["columns"], "final_model_test"), (ENRICHED_COLS, "diff_only_model_test")):
            X = prev[cols].values
            m = make_model(METRICS["selected_model"]).fit(X[TRAIN_FINAL], np.log(y[TRAIN_FINAL]))
            self.assertAlmostEqual(rmse(y[TEST], np.exp(m.predict(X[TEST]))), METRICS[key]["rmse_ms"], delta=0.1)

    def test_saved_model_matches_retraining(self):
        prev = load_dataset(ROOT, "previous")
        bundle = joblib.load(os.path.join(ROOT, "results", "model.joblib"))
        y = prev["cpu_ms"].values.astype(float)
        p = np.exp(bundle["model"].predict(prev[bundle["columns"]].values[TEST]))
        self.assertAlmostEqual(rmse(y[TEST], p), METRICS["final_model_test"]["rmse_ms"], delta=0.1)

    def test_paired_cv_covers_100_folds(self):
        for fs, e in METRICS["E7_paired_cv_gradient_boosting"].items():
            self.assertEqual(e["n_folds"], 100, fs)


class DocumentsMatchResults(unittest.TestCase):
    """Every headline number quoted in the README must be the one in metrics.json."""

    def setUp(self):
        self.readme = read("README.md")

    def assertQuoted(self, value):
        self.assertIn(value, self.readme)

    def test_headline_numbers(self):
        m = METRICS
        fin, diff = m["final_model_test"], m["diff_only_model_test"]
        for v in (fin["rmse_ms"], fin["mae_ms"], diff["rmse_ms"], diff["mae_ms"],
                  m["E2_baselines"]["mean"]["test"]["rmse_ms"], m["E1_reproduce_paper"]["with_cross"]["test"]["rmse_ms"],
                  m["E1_reproduce_paper"]["paper_test_bug_untrained_model_rmse"]):
            self.assertQuoted(f"{v:,.0f}")
        self.assertQuoted(f"{fin['spearman']:.2f}")
        self.assertQuoted(f"{fin['auc_expensive']:.2f}")
        cv = m["cv5"]["selected"]
        self.assertQuoted(f"{cv['rmse_ms'][0]:,.0f} ± {cv['rmse_ms'][1]:,.0f}")
        e7 = m["E7_paired_cv_gradient_boosting"]["diff+size_before"]
        self.assertQuoted(f"{-e7['mean_diff']['rmsle']:.3f}")
        self.assertQuoted(f"{e7['folds_better']['rmsle']:.0%}")
        e6 = m["E6_proxy_validation"]
        self.assertQuoted(f"{e6['proxy A: fixed formula']['spearman_cpu']:.2f}")
        self.assertQuoted(f"{e6['proxy B: calibrated on javac/g++']['spearman_cpu']:.2f}")

    def test_writeup_is_two_pages(self):
        pdf = read("docs/writeup.pdf", "rb")
        self.assertEqual(len(re.findall(rb"/Type /Page[^s]", pdf)), 2)


class Demo(unittest.TestCase):
    def run_demo(self, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "src", "predict.py"), *args],
                              capture_output=True, text=True, check=True).stdout

    def test_replay_row(self):
        out = self.run_demo("--row", "415")
        self.assertIn("Predicted CPU time", out)
        self.assertIn("Actual CPU time", out)

    def test_file_list_uses_fallback(self):
        out = self.run_demo("--files", "site/docs/install.md")
        self.assertIn("diff-features-only model", out)
        self.assertIn("CHEAP", out)


@unittest.skipUnless(os.environ.get("BAZEL_REPO"), "set BAZEL_REPO to a bazelbuild/bazel clone")
class AgainstGit(unittest.TestCase):
    """Recompute a sample of rows straight from git and compare with the committed data."""

    def test_size_features_and_demo_match_git(self):
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from predict import files_from_commit, size_features_from_commit
        repo = os.environ["BAZEL_REPO"]
        diffs = pd.read_csv(os.path.join(ROOT, "data/processed/commit_diffs.csv"), keep_default_na=False)
        size = pd.read_csv(os.path.join(ROOT, "data/processed/size_features.csv"))
        for i in (0, 57, 233, 407, 499):
            sha = diffs.at[i, "sha"]
            files, add, dele = files_from_commit(repo, sha)
            self.assertEqual("|".join(files), diffs.at[i, "files"], f"row {i}")
            self.assertEqual((add, dele), (diffs.at[i, "lines_added"], diffs.at[i, "lines_deleted"]))
            feats = size_features_from_commit(repo, sha)
            for c in SIZE_COLS["before"]:
                raw = size.at[i, c.replace("log_", "")]
                self.assertAlmostEqual(feats[c], np.log1p(raw), places=9, msg=f"row {i} {c}")


if __name__ == "__main__":
    unittest.main()

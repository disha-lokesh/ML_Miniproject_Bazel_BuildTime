"""
Run every experiment and write results/metrics.json + figures.

    python src/train.py

Experiments
  E1  Reproduce the paper: linear regression (mini-batch SGD, lr=0.01, batch 16)
      on one-hot prefix + type, with and without the feature cross.
  E2  Baselines: always predict the training mean / median.
  E3  Fix the commit-alignment bug, keep everything else the same.
  E4  Enriched diff features (aligned) + models on log(CPU time).
  E5  + size and dependency features of the changed files (bytes, imports/#includes).
      The feature set and the model are chosen together on validation RMSLE.
  E6  Do file-based cost proxies from the 47k-commit companion project track real CPU time?
  E7  Paired repeated 5-fold CV: does adding the size features help on the same folds?
Split (same as paper): rows 0-299 train, 300-399 validation, 400-499 test.
Model selection uses validation only; test is reported once at the end.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from features import ENRICHED_COLS, PROXY_COLS, SIZE_COLS, load_dataset  # noqa: E402
from models import SGDLinearRegression, make_model, paper_design_matrix, rmse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

TRAIN, VAL, TEST = np.arange(0, 300), np.arange(300, 400), np.arange(400, 500)
EXPENSIVE_MS = 5000  # "expensive build" = top quartile (> 5 s CPU)

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e8e7e2",
                     "grid.linewidth": 0.6, "figure.dpi": 150})


def evaluate(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1, None)
    return {
        "rmse_ms": round(rmse(y, p), 1),
        "mae_ms": round(float(np.mean(np.abs(y - p))), 1),
        "rmsle": round(float(np.sqrt(np.mean((np.log(y) - np.log(p)) ** 2))), 3),
        "spearman": round(float(spearmanr(y, p)[0]), 3) if np.ptp(p) > 0 else 0.0,
        "auc_expensive": round(float(roc_auc_score(y > EXPENSIVE_MS, p)), 3) if np.ptp(p) > 0 else 0.5,
    }


def main():
    same = load_dataset(ROOT, align="same")         # features as in the paper
    prev = load_dataset(ROOT, align="previous")     # corrected alignment
    y = same["cpu_ms"].values.astype(float)
    res = {"n_builds": int(len(y)), "target_summary_ms": {
        k: round(float(v), 1) for k, v in pd.Series(y).describe().items()}}

    # ---------- E1: reproduce the paper ----------
    e1 = {}
    for cross in (False, True):
        X = paper_design_matrix(same, cross=cross).values
        m = SGDLinearRegression(lr=0.01, batch_size=16).fit(X[TRAIN], y[TRAIN], X[VAL], y[VAL])
        key = "with_cross" if cross else "no_cross"
        e1[key] = {"train": evaluate(y[TRAIN], m.predict(X[TRAIN])),
                   "val": evaluate(y[VAL], m.predict(X[VAL])),
                   "test": evaluate(y[TEST], m.predict(X[TEST])),
                   "history": [[round(a, 1), round(b, 1)] for a, b in m.history]}
    # the paper's test_model() built a NEW, untrained estimator -> predicts 0
    e1["paper_test_bug_untrained_model_rmse"] = round(rmse(y[TEST], np.zeros(len(TEST))), 1)
    res["E1_reproduce_paper"] = e1

    # ---------- E2: baselines ----------
    res["E2_baselines"] = {
        "mean": {s: evaluate(y[idx], np.full(len(idx), y[TRAIN].mean()))
                 for s, idx in [("val", VAL), ("test", TEST)]},
        "median": {s: evaluate(y[idx], np.full(len(idx), np.median(y[TRAIN])))
                   for s, idx in [("val", VAL), ("test", TEST)]},
    }

    # ---------- E3: does the alignment matter? (single-feature correlations) ----------
    corr = {}
    for col in ["touches_src_main", "only_tests_or_docs", "log_n_files",
                "area_src/main/java", "touches_build_files", "area_site"]:
        corr[col] = {"paper_alignment": round(float(spearmanr(same[col], y)[0]), 3),
                     "fixed_alignment": round(float(spearmanr(prev[col], y)[0]), 3)}
    res["E3_feature_target_spearman"] = corr

    # ---------- E4: enriched features, model selection on validation ----------
    train_idx = TRAIN[1:]  # drop build 1: cold cache, 45 s, not an incremental build
    e4 = {}
    for align_name, df in [("paper_alignment", same), ("fixed_alignment", prev)]:
        X = df[ENRICHED_COLS].values
        for name in ["ridge", "random_forest", "gradient_boosting"]:
            m = make_model(name).fit(X[train_idx], np.log(y[train_idx]))
            e4[f"{name}|{align_name}"] = {
                "val": evaluate(y[VAL], np.exp(m.predict(X[VAL]))),
                "test": evaluate(y[TEST], np.exp(m.predict(X[TEST]))),
            }
    res["E4_enriched_models"] = e4

    # ---------- E5: size/dependency features; choose feature set + model on validation ----------
    feature_sets = {
        "diff": ENRICHED_COLS,
        "diff+size_before": ENRICHED_COLS + SIZE_COLS["before"],   # state build i compiles
        "diff+size_after": ENRICHED_COLS + SIZE_COLS["after"],     # for comparison only
        "diff+proxies": ENRICHED_COLS + PROXY_COLS,                # for comparison only
    }
    SELECTABLE = ("diff", "diff+size_before")
    e5 = {}
    for fs, cols in feature_sets.items():
        X = prev[cols].values
        for name in ["ridge", "random_forest", "gradient_boosting"]:
            m = make_model(name).fit(X[train_idx], np.log(y[train_idx]))
            e5[f"{name}|{fs}"] = {
                "val": evaluate(y[VAL], np.exp(m.predict(X[VAL]))),
                "test": evaluate(y[TEST], np.exp(m.predict(X[TEST]))),
            }
    res["E5_size_features"] = e5
    best = min((k for k in e5 if k.split("|")[1] in SELECTABLE), key=lambda k: e5[k]["val"]["rmsle"])
    best_name, best_fs = best.split("|")
    best_cols = feature_sets[best_fs]
    res["selected_model"] = best_name
    res["selected_features"] = best_fs

    # 5-fold CV on all 499 incremental builds, for a less noisy estimate
    Xp, idx_all = prev[best_cols].values, np.arange(1, 500)
    Xpaper = paper_design_matrix(same).values
    cv = {"selected": [], "paper_features_sgd": [], "mean_baseline": []}
    for tr, te in KFold(5, shuffle=True, random_state=0).split(idx_all):
        tr, te = idx_all[tr], idx_all[te]
        m = make_model(best_name).fit(Xp[tr], np.log(y[tr]))
        cv["selected"].append(evaluate(y[te], np.exp(m.predict(Xp[te]))))
        s = SGDLinearRegression().fit(Xpaper[tr], y[tr])
        cv["paper_features_sgd"].append(evaluate(y[te], s.predict(Xpaper[te])))
        cv["mean_baseline"].append(evaluate(y[te], np.full(len(te), y[tr].mean())))
    res["cv5"] = {k: {met: [round(float(np.mean([f[met] for f in v])), 3),
                            round(float(np.std([f[met] for f in v])), 3)]
                      for met in v[0]} for k, v in cv.items()}

    # ---------- E6: do the cost proxies track real CPU time? (rows 1-499) ----------
    bep = pd.read_csv(os.path.join(ROOT, "data/processed/bep_metrics.csv"))
    signals = {"proxy A: fixed formula": "log_proxy_cost_a",
               "proxy B: calibrated on javac/g++": "log_proxy_time_b",
               "bytes of changed files (before)": "log_before_sum_bytes",
               "imports/#includes (before)": "log_before_sum_deps",
               "files in src/main/java": "area_src/main/java",
               "# files changed": "log_n_files"}
    res["E6_proxy_validation"] = {
        label: {"spearman_cpu": round(float(spearmanr(prev[col][1:], y[1:])[0]), 3),
                "spearman_actions": round(float(spearmanr(prev[col][1:], bep.actions_executed[1:])[0]), 3)}
        for label, col in signals.items()}

    # ---------- E7: paired repeated CV, same folds for both feature sets ----------
    e7 = {}
    for fs in ("diff+size_before", "diff+size_after"):
        diffs = []
        for rep in range(20):
            for tr, te in KFold(5, shuffle=True, random_state=100 + rep).split(idx_all):
                tr, te = idx_all[tr], idx_all[te]
                pair = []
                for cols in (feature_sets["diff"], feature_sets[fs]):
                    X = prev[cols].values
                    m = make_model("gradient_boosting", rep).fit(X[tr], np.log(y[tr]))
                    pair.append(evaluate(y[te], np.exp(m.predict(X[te]))))
                diffs.append({k: pair[1][k] - pair[0][k] for k in ("rmse_ms", "rmsle", "spearman")})
        d = pd.DataFrame(diffs)
        per_rep = d.groupby(np.repeat(np.arange(20), 5)).mean()
        e7[fs] = {
            "mean_diff": {k: round(float(d[k].mean()), 4) for k in d},
            "folds_better": {"rmse_ms": round(float((d.rmse_ms < 0).mean()), 3),
                             "rmsle": round(float((d.rmsle < 0).mean()), 3),
                             "spearman": round(float((d.spearman > 0).mean()), 3)},
            "repetitions_better_rmsle": int((per_rep.rmsle < 0).sum()),
            "n_folds": len(d),
        }
    res["E7_paired_cv_gradient_boosting"] = e7

    # final model on train+val, saved for predict.py
    final_idx = np.arange(1, 400)
    final = make_model(best_name).fit(Xp[final_idx], np.log(y[final_idx]))
    # diff-only model as fallback for the demo when file sizes are unknown (--files mode)
    Xd = prev[ENRICHED_COLS].values
    fallback = make_model(best_name).fit(Xd[final_idx], np.log(y[final_idx]))
    joblib.dump({"model": final, "columns": best_cols, "name": best_name, "features": best_fs,
                 "fallback_model": fallback, "fallback_columns": ENRICHED_COLS},
                os.path.join(ROOT, "results", "model.joblib"))
    res["final_model_test"] = evaluate(y[TEST], np.exp(final.predict(Xp[TEST])))
    # the same model on diff features only (the selection before E5 existed), for comparison
    res["diff_only_model_test"] = evaluate(y[TEST], np.exp(fallback.predict(Xd[TEST])))

    # BEP sanity: CPU time is driven by how many actions Bazel re-executes
    res["bep_analysis"] = {
        "spearman_cpu_vs_actions_executed": round(float(spearmanr(bep.cpu_ms, bep.actions_executed)[0]), 3),
        "builds_with_1_action": int((bep.actions_executed == 1).sum()),
    }

    with open(os.path.join(ROOT, "results", "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    make_figures(res, y, prev, bep, final, Xp, e1, best_cols)
    print(json.dumps({k: res[k] for k in ["selected_model", "selected_features", "final_model_test",
                                          "cv5", "E6_proxy_validation", "E7_paired_cv_gradient_boosting"]},
                     indent=2))


def make_figures(res, y, prev, bep, final, Xp, e1, feat_cols):
    # 1. target distribution
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(np.log10(y), bins=40, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("CPU time per build (log10 ms)")
    ax.set_ylabel("builds")
    ax.set_title("Build CPU time is heavily right-skewed", loc="left", fontsize=11)
    ax.axvline(np.log10(np.median(y)), color=INK, lw=1, ls="--")
    ax.text(np.log10(np.median(y)) + 0.03, ax.get_ylim()[1] * 0.9, "median 1.8 s", color=MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "target_distribution.png")); plt.close(fig)

    # 2. CPU vs actions executed
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(bep.actions_executed, bep.cpu_ms, s=10, color=BLUE, alpha=0.6, edgecolor="none")
    ax.set_xscale("symlog"); ax.set_yscale("log")
    ax.set_xlabel("actions Bazel re-executed"); ax.set_ylabel("CPU time (ms)")
    ax.set_title(f"CPU time tracks re-executed actions (Spearman {res['bep_analysis']['spearman_cpu_vs_actions_executed']})",
                 loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cpu_vs_actions.png")); plt.close(fig)

    # 3. alignment fix
    c = res["E3_feature_target_spearman"]
    names = list(c)
    pretty = {"touches_src_main": "touches src/main", "only_tests_or_docs": "only tests/docs",
              "log_n_files": "# files changed", "area_src/main/java": "# files in src/main/java",
              "touches_build_files": "touches BUILD/.bzl", "area_site": "# files in site/"}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    yy = np.arange(len(names))
    ax.barh(yy + 0.2, [c[n]["paper_alignment"] for n in names], 0.38, color=GRAY, label="paper: diff of same commit")
    ax.barh(yy - 0.2, [c[n]["fixed_alignment"] for n in names], 0.38, color=BLUE, label="fixed: diff of previous commit")
    ax.set_yticks(yy, [pretty[n] for n in names]); ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Spearman correlation with CPU time")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Same features, correct commit: signal appears", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "alignment_fix.png")); plt.close(fig)

    # 4. paper reproduction curves
    fig, ax = plt.subplots(figsize=(5, 3))
    h = np.array(e1["with_cross"]["history"])
    ax.plot(h[:, 0], color=BLUE, lw=2, label="training")
    ax.plot(h[:, 1], color=ORANGE, lw=2, label="validation")
    ax.axhline(res["E2_baselines"]["mean"]["val"]["rmse_ms"], color=GRAY, lw=1.2, ls="--", label="predict-the-mean (val)")
    ax.set_ylim(0, 10000); ax.set_xlabel("period"); ax.set_ylabel("RMSE (ms)")
    ax.set_title("Reproduced paper model (with feature cross)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "paper_reproduction.png")); plt.close(fig)

    # 5. test-set model comparison
    rows = [("Untrained (paper's test bug)", rmse(y[TEST], np.zeros(len(TEST))), None),
            ("Predict mean", res["E2_baselines"]["mean"]["test"]["rmse_ms"], 0),
            ("Paper model (SGD + cross)", e1["with_cross"]["test"]["rmse_ms"], e1["with_cross"]["test"]["spearman"]),
            ("Ours, wrong alignment", res["E4_enriched_models"][f"{res['selected_model']}|paper_alignment"]["test"]["rmse_ms"],
             res["E4_enriched_models"][f"{res['selected_model']}|paper_alignment"]["test"]["spearman"]),
            ("Ours, fixed alignment", res["diff_only_model_test"]["rmse_ms"], res["diff_only_model_test"]["spearman"]),
            ("Ours, + size features (final)", res["final_model_test"]["rmse_ms"], res["final_model_test"]["spearman"])]
    fig, ax = plt.subplots(figsize=(6, 3.3))
    cols = [GRAY, GRAY, ORANGE, ORANGE, BLUE, BLUE]
    ax.barh(range(len(rows)), [r[1] for r in rows], color=cols, height=0.6)
    for i, r in enumerate(rows):
        lab = f"{r[1]:,.0f} ms" + (f"   ρ={r[2]:.2f}" if r[2] is not None else "")
        ax.text(r[1] + 80, i, lab, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 9800); ax.set_xlabel("test RMSE (ms, lower is better)")
    ax.set_title("Test set (builds 401-500)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "model_comparison.png")); plt.close(fig)

    # 6. predicted vs actual (final model, test)
    p = np.exp(final.predict(Xp[TEST]))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(y[TEST], p, s=14, color=BLUE, alpha=0.75, edgecolor="none")
    lo, hi = 400, 60000
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual CPU time (ms)"); ax.set_ylabel("predicted CPU time (ms)")
    ax.set_title(f"Final model on test (ρ={res['final_model_test']['spearman']})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pred_vs_actual.png")); plt.close(fig)

    # 7. feature importance (tree models only)
    if hasattr(final, "feature_importances_"):
        imp = pd.Series(final.feature_importances_, index=feat_cols).sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.barh(imp.index.str.replace("area_", "files in ").str.replace("type_", "type ")
                .str.replace("log_", "log "), imp.values, color=BLUE, height=0.6)
        ax.set_xlabel("importance"); ax.set_title("Top 10 features", loc="left", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance.png")); plt.close(fig)

    # 8. proxy validation: Spearman of single signals with real CPU time
    v = res["E6_proxy_validation"]
    labels = sorted(v, key=lambda k: v[k]["spearman_cpu"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = [ORANGE if k.startswith("proxy") else BLUE for k in labels]
    ax.barh(range(len(labels)), [v[k]["spearman_cpu"] for k in labels], color=colors, height=0.6)
    for i, k in enumerate(labels):
        ax.text(v[k]["spearman_cpu"] + 0.01, i, f"{v[k]['spearman_cpu']:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 1); ax.set_xlabel("Spearman correlation with real CPU time")
    ax.set_title("Cost proxies (orange) vs single features", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "proxy_validation.png")); plt.close(fig)

    # 9. paired CV: change in RMSLE when size features are added (negative = better)
    e7 = res["E7_paired_cv_gradient_boosting"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    names = list(e7)
    ax.barh(range(len(names)), [e7[n]["mean_diff"]["rmsle"] for n in names], color=[BLUE, GRAY], height=0.5)
    for i, n in enumerate(names):
        ax.text(e7[n]["mean_diff"]["rmsle"] / 2, i,
                f"{e7[n]['mean_diff']['rmsle']:+.3f}  ({e7[n]['folds_better']['rmsle']:.0%} of folds better)",
                va="center", ha="center", fontsize=8.5, color="white")
    ax.set_yticks(range(len(names)), ["+ size (before state)", "+ size (after state)"])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("change in RMSLE vs diff features (20 x 5-fold CV)")
    ax.set_title("Size/dependency features, paired comparison", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "size_features_paired.png")); plt.close(fig)


if __name__ == "__main__":
    main()

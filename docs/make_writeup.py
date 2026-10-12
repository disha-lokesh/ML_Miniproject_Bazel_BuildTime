"""Builds docs/writeup.pdf (two pages) from results/metrics.json and figures."""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results/metrics.json")))
FIG = os.path.join(ROOT, "results/figures")

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1b1b1b")
MUTED = colors.HexColor("#52514e")

base = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=12, textColor=INK, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=15, leading=18,
                    alignment=TA_CENTER, spaceAfter=2)
sub = ParagraphStyle("sub", parent=base, alignment=TA_CENTER, textColor=MUTED, fontSize=8.5)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    textColor=BLUE, spaceBefore=5, spaceAfter=2)
bul = ParagraphStyle("bul", parent=base, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
cap = ParagraphStyle("cap", parent=base, fontSize=7.5, leading=9, textColor=MUTED, alignment=TA_CENTER)
cell = ParagraphStyle("cell", parent=base, fontSize=8, leading=10, spaceAfter=0)


def P(t, s=base):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, bul, bulletText="•")


def fig(name, w):
    from PIL import Image as PI
    im = PI.open(os.path.join(FIG, name))
    return Image(os.path.join(FIG, name), width=w, height=w * im.height / im.width)


e1 = R["E1_reproduce_paper"]
mean = R["E2_baselines"]["mean"]["test"]
sel = R["selected_model"]
wrong = R["E4_enriched_models"][f"{sel}|paper_alignment"]["test"]
fin = R["final_model_test"]
cv = R["cv5"]
c = R["E3_feature_target_spearman"]
t = R["target_summary_ms"]
e7 = R["E7_paired_cv_gradient_boosting"]["diff+size_before"]

figs0 = Table([[fig("proxy_validation.png", 72 * mm), fig("feature_importance.png", 62 * mm)],
               [P("Fig A. Spearman correlation with real CPU time: cost proxies (orange) vs single features.", cap),
                P("Fig B. Final model, gradient-boosting feature importance.", cap)]],
              colWidths=[90 * mm, 88 * mm])
figs0.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))

story = [
    P("Predicting Bazel Build Times with Machine Learning", h1),
    P("UE24CS352A Machine Learning, Mini-Project &nbsp;|&nbsp; Team: Deepak &amp; Manaswin "
      "&nbsp;|&nbsp; PES University, Dept. of CSE (AI-ML)", sub),
    Spacer(1, 4),

    P("1. Problem statement", h2),
    P("Bazel is Google's open-source build system. Because it is <b>incremental</b>, a new commit only "
      "re-executes the build actions whose inputs changed, so build cost varies from under a second to "
      "tens of seconds. The task, taken from Nadeem &amp; Raza (Stanford CS229), is to predict the "
      "<b>CPU time of the build that follows a commit</b> using only information available before "
      "building: which files the commit changed. Such a predictor lets CI schedulers batch or prioritise "
      "builds and warns developers that a change will be expensive. Our goal was to reproduce the paper, "
      "audit it, and improve it."),

    P("2. Dataset", h2),
    P(f"500 consecutive commits of <font face='Courier'>bazelbuild/bazel</font> (May–June 2021). The "
      f"authors checked out each commit with <font face='Courier'>git checkout HEAD~1</font>, ran "
      f"<font face='Courier'>bazel build //src:bazel-dev</font> and read <i>cpuTimeInMs</i> from the Build "
      f"Event Protocol (BEP) JSON. Target: CPU ms, median {t['50%']/1000:.1f} s, mean {t['mean']/1000:.1f} s, "
      f"max {t['max']/1000:.1f} s, heavily right-skewed. Paper inputs: the single most common path prefix "
      f"(11 values) and file type (5 values) per commit. <b>Our enrichment:</b> the stored commit IDs match "
      f"Bazel's first-parent history one-to-one, so we recovered the <b>full list of changed files and "
      f"line counts</b> for all 500 commits, and extracted <i>actionsExecuted</i> from the BEP files. "
      f"Split as in the paper: builds 1–300 train, 301–400 validation, 401–500 test."),

    P("3. Approach", h2),
    B("<b>Reproduce.</b> Linear regression trained by mini-batch SGD (lr 0.01, batch 16, our numpy "
      "implementation) on one-hot prefix + type and their 55-way <b>feature cross</b>; compared with a "
      "predict-the-mean baseline."),
    B("<b>Audit.</b> Read the original code and data. Found (i) <font face='Courier'>test_model()</font> "
      "builds a new <i>untrained</i> regressor, so all test predictions are 0 (RMSE "
      f"{e1['paper_test_bug_untrained_model_rmse']:,.0f} ms, the flat test line in the paper); (ii) "
      "substring matching: <font face='Courier'>'.c'</font> matches .cc/.css, <font face='Courier'>'.h'</font> "
      "matches .html, commits with no listed type default to JAVA, and <font face='Courier'>'tools/'</font> "
      "matches every <font face='Courier'>com/google/devtools/</font> path (144 of 171 'tools/' labels have no "
      "file in tools/)."),
    B("<b>Fix the alignment.</b> Walking history backwards with a warm cache means build <i>i</i> re-does "
      "the work caused by the difference between commit <i>i</i> and the previously built commit <i>i−1</i>: "
      "the files changed <b>by commit i−1</b>. The paper used commit <i>i</i>'s own diff. We shift features by one."),
    B("<b>Richer features.</b> Files are mapped to a source area (src/main/java, src/test, site, …, by longest "
      "prefix) and a type (by real extension). Features: per-area and per-type counts, log #files, log lines "
      "added/deleted, flags <i>touches src/main</i>, <i>only tests/docs</i>, <i>touches BUILD/.bzl</i>: 32 in all."),
    B("<b>Size and dependency features.</b> For the changed files: total and largest size in bytes and number "
      "of imports / <font face='Courier'>#include</font>s, overall and for Java and C/C++ (8 more features), "
      "read from git in the state <i>before</i> the commit, the version the backwards walk compiles."),
    B("<b>Proxy check.</b> Two file-based cost estimates from a companion project that mined all 47k Bazel "
      "commits (fixed language/directory weights; and compile times fitted on measured <i>javac</i>/<i>g++</i> "
      "runs) are compared with the real CPU time, and tried as features."),
    B("<b>Models.</b> Ridge regression, random forest and gradient boosting (Huber loss) on <b>log CPU time</b>; "
      f"feature set and model selected together on validation RMSLE ({sel.replace('_', ' ')} with size "
      "features won); final model retrained on train+validation and evaluated once on test. Because 100 test "
      "builds are few, feature sets are also compared by paired 5-fold CV repeated 20 times. The cold-cache "
      "first build (44.9 s) is excluded from training."),
]

# ---- page 2 ----
fw = 86 * mm
figs = Table([[fig("alignment_fix.png", fw), fig("model_comparison.png", fw)],
              [P("Fig 1. Spearman correlation of features with CPU time, before and after the alignment fix.", cap),
               P("Fig 2. Test RMSE; ρ = Spearman rank correlation.", cap)]],
             colWidths=[fw + 3 * mm, fw + 3 * mm])
figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))

rows = [["Model (test, builds 401–500)", "RMSE ms", "MAE ms", "RMSLE", "Spearman", "AUC >5 s"],
        ["Predict training mean", f"{mean['rmse_ms']:,.0f}", f"{mean['mae_ms']:,.0f}", f"{mean['rmsle']:.2f}", "0.00", "0.50"]]
for name, m in [("Paper model reproduced (SGD + cross)", e1["with_cross"]["test"]),
                ("Diff features, paper alignment", wrong),
                ("Diff features, fixed alignment", R["diff_only_model_test"]),
                ("+ size features (final, chosen on validation)", fin)]:
    rows.append([name, f"{m['rmse_ms']:,.0f}", f"{m['mae_ms']:,.0f}", f"{m['rmsle']:.2f}",
                 f"{m['spearman']:.2f}", f"{m['auc_expensive']:.2f}"])
tbl = Table(rows, colWidths=[70 * mm, 21 * mm, 21 * mm, 19 * mm, 21 * mm, 21 * mm])
tbl.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Helvetica", 8.2),
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.2),
    ("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 8.2),
    ("TEXTCOLOR", (0, -1), (-1, -1), BLUE),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, MUTED),
    ("LINEBELOW", (0, -1), (-1, -1), 0.6, MUTED),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
]))

fw2 = 70 * mm
figs2 = Table([[fig("pred_vs_actual.png", 58 * mm), fig("size_features_paired.png", 100 * mm)],
               [P("Fig 3. Final model, test builds (log axes). Cheap builds are nailed; the largest are ranked "
                  "correctly but under-predicted.", cap),
                P("Fig 4. Paired 20 x 5-fold CV: change in RMSLE when size features are added (negative is "
                  "better). Before and after state give the same gain.", cap)]],
              colWidths=[72 * mm, 106 * mm])
figs2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))

story += [
    P("4. Implementation", h2),
    P("Python 3 with pandas, scikit-learn and matplotlib (the original TensorFlow-1 Estimator code no longer "
      "runs on TF 2). <font face='Courier'>scripts/enrich_from_git.py</font> rebuilds the diff data and "
      "<font face='Courier'>scripts/size_features.py</font> the size features from a Bazel clone; "
      "<font face='Courier'>src/features.py</font> holds path classification and the "
      "alignment; <font face='Courier'>src/models.py</font> the SGD regressor and model zoo; "
      "<font face='Courier'>src/train.py</font> runs every experiment in about 4.5 min and writes metrics.json, "
      "figures and the model; <font face='Courier'>src/predict.py</font> is the demo CLI (changed files, "
      "a real commit, or a replayed dataset row → predicted CPU time and a cheap/medium/expensive label)."),
    Spacer(1, 4),
    figs0,
    PageBreak(),
    P("5. Results", h2),
    figs,
    Spacer(1, 3),
    tbl,
    Spacer(1, 4),
    P(f"The reproduced paper model matches the paper's training RMSE (about 5.4 to 5.7 s) but is <b>no better than "
      f"predicting the mean</b> (ρ = {e1['with_cross']['test']['spearman']:.2f}). Our richer features alone do "
      f"not help (ρ = {wrong['spearman']:.2f}); the signal appears only after the alignment fix "
      f"(<i>touches src/main</i>: ρ {c['touches_src_main']['paper_alignment']:.2f} → "
      f"{c['touches_src_main']['fixed_alignment']:.2f}). The final model cuts MAE by "
      f"{100 * (1 - fin['mae_ms'] / e1['with_cross']['test']['mae_ms']):.0f}% and identifies builds over 5 s with "
      f"AUC {fin['auc_expensive']:.2f}. On these 100 test builds it is slightly behind the diff-only model, well "
      f"inside the fold-to-fold spread; 5-fold CV over all 499 incremental builds gives RMSE "
      f"{cv['selected']['rmse_ms'][0]:,.0f} ± {cv['selected']['rmse_ms'][1]:,.0f} ms, ρ "
      f"{cv['selected']['spearman'][0]:.2f} (diff features only: 4,709 ms, 0.78; paper's features: ρ "
      f"{abs(cv['paper_features_sgd']['spearman'][0]):.2f}). On identical folds the size features lower RMSLE by "
      f"{-e7['mean_diff']['rmsle']:.3f} in {e7['folds_better']['rmsle']:.0%} of 100 folds (Fig 4). "
      f"The BEP data explains why paths work: CPU time correlates "
      f"{R['bep_analysis']['spearman_cpu_vs_actions_executed']:.2f} (Spearman) with the number of actions Bazel "
      f"re-executed, and {R['bep_analysis']['builds_with_1_action']} builds ran a single action (docs/test-only changes)."),

    figs2,
    P("6. Conclusions", h2),
    B("Build cost is driven by <b>how much of the dependency graph a change invalidates</b>; which files "
      "changed (core <font face='Courier'>src/main/java</font> vs tests or docs) is a strong proxy for that, "
      "and how large the changed files are and how many dependencies they declare adds a consistent gain."),
    B(f"Hand-built cost formulas are weaker than learned features: the two proxies reach ρ "
      f"{R['E6_proxy_validation']['proxy A: fixed formula']['spearman_cpu']:.2f} and "
      f"{R['E6_proxy_validation']['proxy B: calibrated on javac/g++']['spearman_cpu']:.2f}; measured single-file "
      "compile times do not transfer to what Bazel re-runs."),
    B("Data provenance mattered more than the model: an off-by-one between features and labels erased all "
      "signal. Always compare against a trivial baseline and look at rank metrics, not RMSE alone, when the "
      "target is skewed."),
    B("Limitations: 500 builds, one machine and one target; 100 test builds are too few to separate models "
      "a few percent apart; the model compresses the extreme tail "
      "(> 15 s builds are ranked correctly but under-predicted). Next step: reverse-dependency counts from "
      "<font face='Courier'>bazel query 'rdeps(...)'</font> as features, and more commits."),
    P("<i>Reference:</i> M. U. Nadeem, S. Raza, \"Predicting Bazel Build Times Using Machine Learning\", "
      "Stanford CS229; code and data: github.com/umar-nadeem/CS229-Final-Project.", cap),
]

doc = SimpleDocTemplate(os.path.join(ROOT, "docs", "writeup.pdf"), pagesize=A4,
                        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm, bottomMargin=12 * mm,
                        title="Predicting Bazel Build Times with ML: Write-up", author="Deepak, Manaswin")
doc.build(story)
print("ok")

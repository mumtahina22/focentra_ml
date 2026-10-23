"""
paper_eval.py - extra evaluation for the Focentra paper.

Run it from the focentra_ml folder (the same place as train_global_model.py):

    venv\\Scripts\\activate
    python paper_eval.py

It reads
    data/processed/training_data.csv
    app/models/global_model.pkl
    app/models/label_encoder.pkl
    app/features/cold_start.py        (the production cold-start code)

and writes everything into a new folder, paper_results/:
    results.md                 tables for the paper (send this file to Claude)
    results.json               the same numbers, machine-readable
    fig_baselines.png          Fig: model comparison
    fig9_calibration.png        Fig: reliability diagram of the deployed model
    fig8_coldstart_evidence.png Fig: accuracy as behavioural evidence accumulates

Nothing is written to Supabase and the deployed model files are not changed.
Takes about 1-3 minutes on a laptop.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.features.cold_start import build_cold_start_vector

# Same order as app/features/extractor.py FEATURE_COLUMNS (copied here so this
# script does not need a Supabase connection).
FEATURE_COLUMNS = [
    "completion_ratio_daily",
    "completion_ratio_weekly",
    "completion_ratio_monthly",
    "focus_sessions_per_day",
    "avg_session_hour",
    "session_hour_variance",
    "consistency_score",
    "streak_recovery_rate",
    "motivation_curve_slope",
    "points_burst_ratio",
    "weekly_point_delta",
    "task_creation_rate",
    "onboarding_c",
    "onboarding_n",
    "onboarding_o",
]
ONBOARDING = ["onboarding_c", "onboarding_n", "onboarding_o"]

DATA_PATH = Path("data/processed/training_data.csv")
MODEL_PATH = Path("app/models/global_model.pkl")
ENCODER_PATH = Path("app/models/label_encoder.pkl")
OUT = Path("paper_results")
OUT.mkdir(exist_ok=True)

SEED = 42

# Figure style: white surface, dark ink, hairline grid, thin marks.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # highlighted / series 1
ORANGE = "#eb6834"   # series 2
GRAY = "#b9b8b1"     # de-emphasised bars / reference lines

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")


# ─── 1. Data and the exact split used in train_global_model.py ─────────────
print("Loading data and the deployed model...")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

X = df[FEATURE_COLUMNS].values
y = encoder.transform(df["archetype"].values)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Train {len(X_train)} rows | Test {len(X_test)} rows")

results = {"n_train": int(len(X_train)), "n_test": int(len(X_test))}

proba_test = model.predict_proba(X_test)
pred_test = proba_test.argmax(axis=1)
results["deployed_model"] = {
    "accuracy": round(accuracy_score(y_test, pred_test), 4),
    "macro_f1": round(macro_f1(y_test, pred_test), 4),
    "weighted_f1": round(weighted_f1(y_test, pred_test), 4),
}
print("Deployed model:", results["deployed_model"])


# ─── 2. Baselines on the same split ───────────────────────────────────────
print("\nTraining baselines (same split)...")
baselines = [
    ("Majority class", DummyClassifier(strategy="most_frequent")),
    ("Gaussian naive Bayes", GaussianNB()),
    ("Logistic regression",
     make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
    ("Decision tree (depth 8)",
     DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                            class_weight="balanced", random_state=SEED)),
    ("Random forest (deployed settings)",
     RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                            class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ("Gradient boosting (HistGB)",
     HistGradientBoostingClassifier(random_state=SEED)),
    ("Decision tree (unconstrained)",
     DecisionTreeClassifier(random_state=SEED)),
    ("Random forest (unconstrained)",
     RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
]
baseline_rows = []
for name, clf in baselines:
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    p = clf.predict(X_test)
    row = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, p), 4),
        "macro_f1": round(macro_f1(y_test, p), 4),
        "fit_seconds": round(fit_s, 2),
    }
    baseline_rows.append(row)
    print(f"  {name:<36} acc {row['accuracy']:.4f}  macro-F1 {row['macro_f1']:.4f}")
results["baselines"] = baseline_rows


# ─── 3. Feature ablation (deployed RF settings) ───────────────────────────
print("\nAblation...")
ablations = [
    ("All 15 features", []),
    ("Without onboarding_c/n/o", ONBOARDING),
    ("Without avg_session_hour", ["avg_session_hour"]),
    ("Without onboarding and avg_session_hour", ONBOARDING + ["avg_session_hour"]),
]
ablation_rows = []
for name, drop in ablations:
    keep = [i for i, c in enumerate(FEATURE_COLUMNS) if c not in drop]
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(X_train[:, keep], y_train)
    p = clf.predict(X_test[:, keep])
    row = {
        "variant": name,
        "n_features": len(keep),
        "accuracy": round(accuracy_score(y_test, p), 4),
        "macro_f1": round(macro_f1(y_test, p), 4),
    }
    ablation_rows.append(row)
    print(f"  {name:<42} acc {row['accuracy']:.4f}  macro-F1 {row['macro_f1']:.4f}")
results["ablation"] = ablation_rows


# ─── 4. Calibration of the deployed model ─────────────────────────────────
print("\nCalibration...")
conf = proba_test.max(axis=1)
correct = (pred_test == y_test).astype(float)
bins = np.linspace(0, 1, 11)
bin_rows = []
ece = 0.0
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (conf > lo) & (conf <= hi)
    n = int(mask.sum())
    if n == 0:
        continue
    acc_b = float(correct[mask].mean())
    conf_b = float(conf[mask].mean())
    ece += (n / len(conf)) * abs(acc_b - conf_b)
    bin_rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": n,
                     "mean_confidence": round(conf_b, 4), "accuracy": round(acc_b, 4)})
onehot = np.eye(proba_test.shape[1])[y_test]
brier = float(np.mean(np.sum((proba_test - onehot) ** 2, axis=1)))
gate = conf >= 0.65
results["calibration"] = {
    "ece_10_bins": round(ece, 4),
    "brier_multiclass": round(brier, 4),
    "mean_confidence": round(float(conf.mean()), 4),
    "share_conf_ge_0_65": round(float(gate.mean()), 4),
    "accuracy_when_conf_ge_0_65": round(float(correct[gate].mean()), 4) if gate.any() else None,
    "accuracy_when_conf_lt_0_65": round(float(correct[~gate].mean()), 4) if (~gate).any() else None,
    "bins": bin_rows,
}
print("  ", {k: v for k, v in results["calibration"].items() if k != "bins"})

fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, zorder=1)
xs = [b["mean_confidence"] for b in bin_rows]
ys = [b["accuracy"] for b in bin_rows]
ax.plot(xs, ys, color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
ax.scatter(xs, ys, s=28, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
ax.axvline(0.65, color=GRAY, linewidth=1, zorder=1)
ax.text(0.655, 0.04, "reassignment\nthreshold 0.65", color=INK_2, fontsize=7, va="bottom")
ax.text(0.30, 0.24, "perfect calibration", color=INK_2, fontsize=7, va="top")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.03)
ax.set_xlabel("Predicted confidence (top class)")
ax.set_ylabel("Observed accuracy")
ax.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title(f"Deployed model calibration (ECE = {ece:.3f})", fontsize=8.5, color=INK)
fig.savefig(OUT / "fig9_calibration.png")
plt.close(fig)


# ─── 5. Cold start: accuracy as behavioural evidence accumulates ──────────
# Uses the production build_cold_start_vector(): population means, the
# onboarding shift, then real values for the feature groups already "observed".
print("\nCold-start evidence curve (production cold-start code)...")
STAGES = [
    ("Onboarding\nonly", []),
    ("+ task\ncompletion", ["completion_ratio_daily", "completion_ratio_weekly",
                           "completion_ratio_monthly", "task_creation_rate"]),
    ("+ focus\nsessions", ["focus_sessions_per_day", "avg_session_hour",
                          "session_hour_variance"]),
    ("+ points\ntrends", ["motivation_curve_slope", "points_burst_ratio",
                         "weekly_point_delta"]),
    ("+ streak\nsignals", ["consistency_score", "streak_recovery_rate"]),
]
test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)


def stage_accuracy(revealed, use_onboarding):
    rows = []
    for _, r in test_df.iterrows():
        c, n, o = ((r["onboarding_c"], r["onboarding_n"], r["onboarding_o"])
                   if use_onboarding else (0.5, 0.5, 0.5))
        partial = {k: float(r[k]) for k in revealed}
        vec = build_cold_start_vector(onboarding_c=float(c), onboarding_n=float(n),
                                      onboarding_o=float(o), partial_features=partial)
        rows.append([vec[col] for col in FEATURE_COLUMNS])
    pr = model.predict_proba(np.array(rows))
    p = pr.argmax(axis=1)
    return {
        "accuracy": round(accuracy_score(y_test, p), 4),
        "macro_f1": round(macro_f1(y_test, p), 4),
        "mean_confidence": round(float(pr.max(axis=1).mean()), 4),
        "share_provisional_lt_0_45": round(float((pr.max(axis=1) < 0.45).mean()), 4),
    }


revealed = []
curve = []
for label, group in STAGES:
    revealed = revealed + group
    with_onb = stage_accuracy(revealed, True)
    neutral = stage_accuracy(revealed, False)
    curve.append({"stage": label.replace("\n", " "),
                  "with_onboarding": with_onb, "neutral_onboarding": neutral})
    print(f"  {label.replace(chr(10), ' '):<22} with onboarding {with_onb['accuracy']:.4f}"
          f" | neutral {neutral['accuracy']:.4f}")
majority_acc = float(np.bincount(y_test).max() / len(y_test))
results["coldstart_curve"] = curve
results["majority_class_accuracy"] = round(majority_acc, 4)

fig, ax = plt.subplots(figsize=(3.5, 2.6))
x = np.arange(len(STAGES))
a1 = [c["with_onboarding"]["accuracy"] for c in curve]
a2 = [c["neutral_onboarding"]["accuracy"] for c in curve]
ax.axhline(majority_acc, color=GRAY, linewidth=1, zorder=1)
ax.text(len(STAGES) - 1, majority_acc + 0.015, f"majority class {majority_acc:.0%}",
        color=INK_2, fontsize=7, ha="right")
ax.plot(x, a1, color=BLUE, linewidth=2, label="Onboarding answers", zorder=3)
ax.scatter(x, a1, s=26, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
ax.plot(x, a2, color=ORANGE, linewidth=2, label="Neutral answers (0.5)", zorder=2)
ax.scatter(x, a2, s=26, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=4)
ax.text(x[0] + 0.08, a1[0], f"{a1[0]:.0%}", color=INK, fontsize=7, va="bottom")
ax.text(x[0] + 0.08, a2[0], f"{a2[0]:.0%}", color=INK, fontsize=7, va="top")
ax.text(x[-1] + 0.08, a1[-1], f"{a1[-1]:.0%}", color=INK, fontsize=7, va="center")
ax.set_xticks(x)
ax.set_xticklabels([s for s, _ in STAGES], fontsize=7)
ax.set_xlim(-0.3, len(STAGES) - 0.55)
ax.set_ylim(0, 1)
ax.set_ylabel("Test accuracy")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(True, axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, handlelength=1.6)
fig.savefig(OUT / "fig8_coldstart_evidence.png")
plt.close(fig)


# ─── 6. Baseline figure ───────────────────────────────────────────────────
rows_sorted = sorted(baseline_rows, key=lambda r: r["macro_f1"])
fig, ax = plt.subplots(figsize=(3.5, 2.6))
names = [r["model"] for r in rows_sorted]
vals = [r["macro_f1"] for r in rows_sorted]
colors = [BLUE if "deployed" in n else GRAY for n in names]
ax.barh(names, vals, color=colors, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7, color=INK)
ax.set_xlim(0, 1.12)
ax.set_xlabel("Macro-F1 on the 4,000-row test set")
ax.grid(True, axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=7)
fig.savefig(OUT / "fig_baselines.png")
plt.close(fig)


# ─── 7. Write results ─────────────────────────────────────────────────────
(OUT / "results.json").write_text(json.dumps(results, indent=2))

md = ["# Focentra paper - evaluation results", ""]
d = results["deployed_model"]
md += [f"Deployed model on the test set: accuracy {d['accuracy']}, "
       f"macro-F1 {d['macro_f1']}, weighted-F1 {d['weighted_f1']}.", ""]
md += ["## Baselines (same 80/20 split)", "",
       "| Model | Accuracy | Macro-F1 | Fit time (s) |", "| --- | --- | --- | --- |"]
md += [f"| {r['model']} | {r['accuracy']} | {r['macro_f1']} | {r['fit_seconds']} |"
       for r in baseline_rows]
md += ["", "## Ablation (deployed RF settings)", "",
       "| Variant | Features | Accuracy | Macro-F1 |", "| --- | --- | --- | --- |"]
md += [f"| {r['variant']} | {r['n_features']} | {r['accuracy']} | {r['macro_f1']} |"
       for r in ablation_rows]
c = results["calibration"]
md += ["", "## Calibration (deployed model)", "",
       f"ECE (10 bins) {c['ece_10_bins']}; multi-class Brier {c['brier_multiclass']}; "
       f"mean confidence {c['mean_confidence']}.",
       f"{c['share_conf_ge_0_65']:.1%} of test predictions reach confidence >= 0.65; "
       f"their accuracy is {c['accuracy_when_conf_ge_0_65']} "
       f"(vs {c['accuracy_when_conf_lt_0_65']} below 0.65).", "",
       "| Confidence bin | n | Mean confidence | Accuracy |", "| --- | --- | --- | --- |"]
md += [f"| {b['bin']} | {b['n']} | {b['mean_confidence']} | {b['accuracy']} |"
       for b in c["bins"]]
md += ["", "## Cold-start evidence curve", "",
       f"Majority-class accuracy: {results['majority_class_accuracy']}.", "",
       "| Stage | Acc. (answers) | Acc. (neutral) | Provisional share (answers) |",
       "| --- | --- | --- | --- |"]
md += [f"| {r['stage']} | {r['with_onboarding']['accuracy']} | "
       f"{r['neutral_onboarding']['accuracy']} | "
       f"{r['with_onboarding']['share_provisional_lt_0_45']} |" for r in curve]
(OUT / "results.md").write_text("\n".join(md), encoding="utf-8")

print(f"\nDone. Results and figures are in {OUT.resolve()}")
print("Send paper_results/results.md to Claude to fill in the paper's tables.")

"""
train_global_model.py  (version 3)

Trains the global archetype classifier and saves it to app/models/global_model.pkl.

Version 3 change (after review): the Random Forest keeps its settings but
uses the 12 behavioural features only. The three onboarding answers are no
longer model inputs, because in the training data they were computed from
behaviour (onboarding_n is the same column as streak_recovery_rate), while in
the app they are self-reports. They now act only through the cold-start shift.

Run with:  python train_global_model.py      (takes about 1-2 minutes)
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── PATHS ───────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed/training_data.csv")
MODEL_DIR = Path("app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "global_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

# ─── FEATURE COLUMNS (must match extractor.py FEATURE_COLUMNS exactly) ───────
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
]

SEED = 42


def make_model():
    """Same Random Forest settings as version 1 (now on 12 features)."""
    return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading processed training data...")
df = pd.read_csv(PROCESSED_PATH)
X = df[FEATURE_COLUMNS].values
y_raw = df["archetype"].values
print(f"Rows: {len(df)}, features: {len(FEATURE_COLUMNS)}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Classes: {list(le.classes_)}")

# Same split as version 1 and paper_eval.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\nTraining Random Forest (12 features)...")
t0 = time.perf_counter()
model = make_model()
model.fit(X_train, y_train)
print(f"Training complete in {time.perf_counter() - t0:.0f} s.")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
proba = model.predict_proba(X_test)
y_pred = proba.argmax(axis=1)
print("\n=== TEST SET PERFORMANCE ===")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

print("=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

conf = proba.max(axis=1)
correct = (y_pred == y_test).astype(float)
ece = 0.0
for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.any():
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
print(f"\nCalibration: ECE (10 bins) = {ece:.4f}, mean confidence = {conf.mean():.4f}, "
      f"accuracy = {correct.mean():.4f}")

print("\n=== 5-FOLD CROSS VALIDATION (weighted F1) ===")
print("Training 5 more models, one per fold...")
cv_scores = cross_val_score(make_model(), X, y, cv=5, scoring="f1_weighted", n_jobs=-1)
print(f"F1 scores: {cv_scores.round(4)}")
print(f"Mean F1:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── FEATURE IMPORTANCE (permutation, on the test set) ───────────────────────
print("\n=== PERMUTATION IMPORTANCE (drop in macro-F1 when a feature is shuffled) ===")
perm = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                              n_repeats=10, random_state=SEED, n_jobs=-1)
importances = {c: round(float(m), 4) for c, m in zip(FEATURE_COLUMNS, perm.importances_mean)}
importances_sd = {c: round(float(s), 4) for c, s in zip(FEATURE_COLUMNS, perm.importances_std)}
for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
    print(f"  {feat:<28} {imp:.4f}  {'█' * int(max(imp, 0) * 100)}")
impurity = {c: round(float(v), 4) for c, v in zip(FEATURE_COLUMNS, model.feature_importances_)}
print("\n=== IMPURITY IMPORTANCE (mean decrease in impurity) ===")
for feat, imp in sorted(impurity.items(), key=lambda kv: -kv[1]):
    print(f"  {feat:<28} {imp:.4f}")
IMPORTANCE_PATH.write_text(json.dumps({"mean": importances, "std": importances_sd,
                                       "impurity": impurity}, indent=2))

# ─── SAVE ─────────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\nModel saved      → {MODEL_PATH}")
print(f"Encoder saved    → {ENCODER_PATH}")
print(f"Importance saved → {IMPORTANCE_PATH}")

print("\nIMPORTANT: Railway must load the model with the same library versions.")
print("Make sure these three lines are in requirements.txt (replace the old ones):")
print(f"  scikit-learn=={sklearn.__version__}")
print(f"  numpy=={np.__version__}")
print(f"  joblib=={joblib.__version__}")
print("\nDone. Global model is ready.")

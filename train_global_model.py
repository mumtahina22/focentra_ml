"""
train_global_model.py

Trains the global Random Forest archetype classifier
on the processed dataset and saves it to app/models/global_model.pkl

Run with:  python train_global_model.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ─── PATHS ───────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed/training_data.csv")
MODEL_DIR      = Path("app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH     = MODEL_DIR / "global_model.pkl"
ENCODER_PATH   = MODEL_DIR / "label_encoder.pkl"

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
    "onboarding_c",
    "onboarding_n",
    "onboarding_o",
]

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading processed training data...")
df = pd.read_csv(PROCESSED_PATH)
print(f"Rows: {len(df)}, Columns: {df.columns.tolist()}")

X = df[FEATURE_COLUMNS].values
y = df["archetype"].values

print(f"\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

# ─── ENCODE LABELS ────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded classes: {list(le.classes_)}")

# ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded   # keeps class balance in both splits
)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# ─── TRAIN MODEL ──────────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",   # handles any class imbalance automatically
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)
model.fit(X_train, y_train)
print("Training complete.")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
print("\n=== TEST SET PERFORMANCE ===")
y_pred = model.predict(X_test)
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df.to_string())

print("\n=== 5-FOLD CROSS VALIDATION ===")
cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring="f1_weighted")
print(f"F1 scores: {cv_scores.round(4)}")
print(f"Mean F1:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── FEATURE IMPORTANCES ──────────────────────────────────────────────────────
print("\n=== FEATURE IMPORTANCES ===")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
importances_sorted = importances.sort_values(ascending=False)
for feat, imp in importances_sorted.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ─── SAVE MODEL + ENCODER ─────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\nModel saved    → {MODEL_PATH}")
print(f"Encoder saved  → {ENCODER_PATH}")
print("\nDone. Global model is ready.")
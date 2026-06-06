"""
build_training_data.py

Runs ONCE (or whenever you retrain globally).
Reads the raw Kaggle dataset, maps columns to Focentra
feature space, generates archetype labels using rule-based
logic, and saves the processed training CSV.

Run with:  python build_training_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── PATHS ───────────────────────────────────────────────────────────────────
RAW_PATH       = Path("data/raw/student_productivity.csv")
PROCESSED_PATH = Path("data/processed/training_data.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─── STEP 1: LOAD ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")


# ─── STEP 2: MAP COLUMNS TO FOCENTRA FEATURE SPACE ───────────────────────────
print("Mapping columns to Focentra features...")

features = pd.DataFrame()

# completion_ratio_daily
# assignments_completed (0-19) normalized to [0,1]
features["completion_ratio_daily"] = (
    df["assignments_completed"] / 19.0
).clip(0, 1)

# completion_ratio_weekly — proxy: attendance_percentage / 100
features["completion_ratio_weekly"] = (
    df["attendance_percentage"] / 100.0
).clip(0, 1)

# completion_ratio_monthly — proxy: final_grade / 100
features["completion_ratio_monthly"] = (
    df["final_grade"] / 100.0
).clip(0, 1)

# focus_sessions_per_day
# study_hours_per_day / 1.5 (assume avg session = 1.5h like Pomodoro blocks)
features["focus_sessions_per_day"] = (
    df["study_hours_per_day"] / 1.5
).clip(0, 12)

# avg_session_hour
# We don't have time-of-day in dataset.
# Approximate from sleep_hours:
#   low sleep (< 5h) → likely night owl → session hour ~21
#   normal sleep (6-8h) → morning/afternoon → session hour ~14
#   high sleep (> 8h) → early riser → session hour ~9
features["avg_session_hour"] = np.where(
    df["sleep_hours"] < 5, 21.0,
    np.where(df["sleep_hours"] > 8, 9.0, 14.5)
)

# session_hour_variance
# High phone usage → inconsistent schedule → high variance
features["session_hour_variance"] = (
    df["phone_usage_hours"] / 12.0 * 6.0
).clip(0.5, 6.0)

# consistency_score
# focus_score (30-99) normalized to [0,1]
features["consistency_score"] = (
    (df["focus_score"] - 30) / (99 - 30)
).clip(0, 1)

# streak_recovery_rate
# stress_level (1-10): high stress → more resets → higher recovery rate
features["streak_recovery_rate"] = (
    df["stress_level"] / 10.0
).clip(0, 1)

# motivation_curve_slope
# productivity_score (0-100): high → rising motivation (positive slope)
# Map: 0-50 → negative slope, 50-100 → positive slope
features["motivation_curve_slope"] = (
    (df["productivity_score"] - 50) / 10.0
).clip(-5, 5)

# points_burst_ratio
# High gaming + social media → bursty behavior pattern
distraction = (df["social_media_hours"] + df["gaming_hours"]) / 14.0
features["points_burst_ratio"] = distraction.clip(0, 1)

# weekly_point_delta
# Rising delta → positive correlation with study hours vs phone hours
features["weekly_point_delta"] = (
    (df["study_hours_per_day"] - df["phone_usage_hours"]) * 3
).clip(-30, 30)

# task_creation_rate
# assignments_completed as proxy (more assignments → more task creation)
features["task_creation_rate"] = (
    df["assignments_completed"] / 19.0 * 7.0
).clip(0, 7)

# onboarding scores — approximated from Big Five proxies in dataset
# conscientiousness: high assignments + low social media
features["onboarding_c"] = (
    (features["completion_ratio_daily"] * 0.6) +
    ((1 - features["points_burst_ratio"]) * 0.4)
).clip(0, 1)

# neuroticism: high stress → high N score
features["onboarding_n"] = (
    df["stress_level"] / 10.0
).clip(0, 1)

# openness: high youtube + variety of activities → high O score
features["onboarding_o"] = (
    (df["youtube_hours"] / 6.0 * 0.5) +
    (df["exercise_minutes"] / 119.0 * 0.5)
).clip(0, 1)

print(f"Feature matrix shape: {features.shape}")
print(features.describe().round(3).to_string())


# ─── STEP 3: GENERATE ARCHETYPE LABELS ───────────────────────────────────────
"""
Label assignment rules.
Each row gets exactly ONE archetype label.
Rules are evaluated in priority order — first match wins.
Priority order chosen to prevent overlap on edge cases.
"""

print("\nGenerating archetype labels...")

ARCHETYPES = [
    "Consistent Achiever",
    "Night Owl Performer",
    "Easily Distracted User",
    "Last-Minute Performer",
    "Burnout-Prone User",
]

def assign_archetype(row: pd.Series) -> str:

    # ── Burnout-Prone User ──────────────────────────────────────────
    if (
        row["streak_recovery_rate"] >= 0.80 and
        row["motivation_curve_slope"] <= -2.0 and
        row["consistency_score"] <= 0.35
    ):
        return "Burnout-Prone User"

    # ── Easily Distracted User ──────────────────────────────────────
    if (
        row["points_burst_ratio"] >= 0.60 and
        row["completion_ratio_daily"] <= 0.40 and
        row["session_hour_variance"] >= 4.0
    ):
        return "Easily Distracted User"

    # ── Night Owl Performer ─────────────────────────────────────────
    # Much stricter — needs late hour AND high completion AND good consistency
    # AND low stress (not burnout) AND low distraction (not distracted)
    if (
        row["avg_session_hour"] >= 21.0 and
        row["completion_ratio_weekly"] >= 0.75 and
        row["consistency_score"] >= 0.55 and
        row["streak_recovery_rate"] <= 0.60 and       # not stressed
        row["points_burst_ratio"] <= 0.55             # not distracted
    ):
        return "Night Owl Performer"

    # ── Last-Minute Performer ───────────────────────────────────────
    if (
        row["points_burst_ratio"] >= 0.50 and
        row["weekly_point_delta"] <= -3.0 and
        row["completion_ratio_monthly"] >= 0.60 and
        row["streak_recovery_rate"] < 0.70
    ):
        return "Last-Minute Performer"

    # ── Consistent Achiever ─────────────────────────────────────────
    if (
        row["consistency_score"] >= 0.60 and
        row["completion_ratio_daily"] >= 0.60 and
        row["motivation_curve_slope"] >= 0.5 and
        row["points_burst_ratio"] <= 0.55
    ):
        return "Consistent Achiever"

    # ── Fallback: score-based assignment ───────────────────────────
    # Night Owl score heavily penalized in fallback so it stops dominating
    scores = {
        "Consistent Achiever": (
            row["consistency_score"] * 0.40 +
            row["completion_ratio_daily"] * 0.40 +
            max(row["motivation_curve_slope"] / 5.0, 0) * 0.20
        ),
        "Night Owl Performer": (
            # Only wins fallback if BOTH hour is late AND weekly completion is strong
            (1.0 if row["avg_session_hour"] >= 21.0 else 0.0) * 0.30 +
            row["completion_ratio_weekly"] * 0.30 +
            row["consistency_score"] * 0.20 +
            (1.0 - row["streak_recovery_rate"]) * 0.20
        ),
        "Easily Distracted User": (
            row["points_burst_ratio"] * 0.40 +
            (1.0 - row["completion_ratio_daily"]) * 0.35 +
            (row["session_hour_variance"] / 6.0) * 0.25
        ),
        "Last-Minute Performer": (
            row["points_burst_ratio"] * 0.35 +
            row["completion_ratio_monthly"] * 0.35 +
            max(-row["weekly_point_delta"] / 30.0, 0) * 0.30
        ),
        "Burnout-Prone User": (
            row["streak_recovery_rate"] * 0.40 +
            max(-row["motivation_curve_slope"] / 5.0, 0) * 0.35 +
            (1.0 - row["consistency_score"]) * 0.25
        ),
    }
    return max(scores, key=scores.get)


features["archetype"] = features.apply(assign_archetype, axis=1)


# ─── STEP 4: CHECK LABEL DISTRIBUTION ────────────────────────────────────────
print("\n=== ARCHETYPE DISTRIBUTION ===")
dist = features["archetype"].value_counts()
print(dist)
print(f"\nTotal labeled: {len(features)}")

pct = features["archetype"].value_counts(normalize=True) * 100
print("\n=== PERCENTAGES ===")
print(pct.round(2))


# ─── STEP 5: SAVE ─────────────────────────────────────────────────────────────
features.to_csv(PROCESSED_PATH, index=False)
print(f"\nSaved processed training data → {PROCESSED_PATH}")
print("Done.")
"""
app/features/cold_start.py

Cold start prior feature vector.
ALL defaults are derived from actual training data population means
computed from the Student Productivity and Behavior Dataset (20K).

This guarantees the cold start vector lands INSIDE the training
distribution rather than in an extrapolation gap.

Key insight from archetype mean analysis:
- streak_recovery_rate is the strongest Burnout signal (0.83 vs 0.44-0.55)
- avg_session_hour is the strongest Night Owl signal (19.79 vs 12-14)
- completion_ratio_daily separates Consistent (0.75) from Distracted (0.22)
- onboarding_n mirrors streak_recovery_rate almost exactly (same source)
- motivation_curve_slope separates Burnout (-1.34) from Consistent (+1.28)
"""

# ─── REAL POPULATION MEANS FROM TRAINING DATA ────────────────────────────────
# Source: data/processed/training_data.csv global means
# Used when a user has zero or minimal activity history

POPULATION_DEFAULTS = {
    "completion_ratio_daily":    0.4997,
    "completion_ratio_weekly":   0.6995,
    "completion_ratio_monthly":  0.7027,
    "focus_sessions_per_day":    3.5030,   # was 0.0 — this was the main problem
    "avg_session_hour":          14.7304,
    "session_hour_variance":     3.1303,
    "consistency_score":         0.4992,   # was 0.0 — pushed toward Burnout
    "streak_recovery_rate":      0.5479,   # was 0.0 — far outside any archetype
    "motivation_curve_slope":    0.0180,   # was 0.0 — acceptable, close to real
    "points_burst_ratio":        0.4994,
    "weekly_point_delta":       -2.9718,   # was 0.0 — real mean is slightly negative
    "task_creation_rate":        3.4978,   # was 0.0 — this was the main problem
    "onboarding_c":              0.5000,
    "onboarding_n":              0.5479,   # matches population streak_recovery mean
    "onboarding_o":              0.4998,
}

# ─── ARCHETYPE CENTROIDS (for research paper reference) ──────────────────────
# These are the true cluster centers learned from training data.
# Used in paper to justify archetype separability.

ARCHETYPE_CENTROIDS = {
    "Burnout-Prone User": {
        "consistency_score":      0.2095,
        "streak_recovery_rate":   0.8303,
        "motivation_curve_slope": -1.3393,
        "onboarding_n":           0.8303,
        "avg_session_hour":       14.5907,
    },
    "Consistent Achiever": {
        "consistency_score":      0.7120,
        "streak_recovery_rate":   0.5526,
        "motivation_curve_slope": 1.2821,
        "completion_ratio_daily": 0.7510,
        "avg_session_hour":       12.6782,
    },
    "Easily Distracted User": {
        "completion_ratio_daily": 0.2229,
        "points_burst_ratio":     0.5634,
        "task_creation_rate":     1.5601,
        "onboarding_c":           0.3084,
        "avg_session_hour":       13.1651,
    },
    "Last-Minute Performer": {
        "points_burst_ratio":        0.5968,
        "weekly_point_delta":       -11.6459,
        "completion_ratio_monthly":  0.8113,
        "streak_recovery_rate":      0.4354,
        "avg_session_hour":          13.7193,
    },
    "Night Owl Performer": {
        "avg_session_hour":       19.7936,
        "completion_ratio_weekly": 0.7453,
        "consistency_score":      0.5486,
        "streak_recovery_rate":   0.4498,
        "motivation_curve_slope": -0.0286,
    },
}


# ─── ONBOARDING SCORE → ARCHETYPE PRIOR SHIFT ────────────────────────────────
# When onboarding answers are available, we shift the default vector
# toward the archetype centroid most consistent with those answers.
#
# Logic derived from Big Five → archetype correlations:
# High C (conscientiousness) → toward Consistent Achiever centroid
# High N (neuroticism)       → toward Burnout-Prone centroid
# High O (openness)          → slight Easily Distracted signal
#
# Shift is partial (weighted by score) so it never fully
# overrides the population prior — it just tilts it.

def _apply_onboarding_shift(
    vector: dict,
    c: float,
    n: float,
    o: float,
) -> dict:
    """
    Shift population defaults toward archetype centroids
    based on onboarding personality proxy scores.

    Shift weights derived from archetype centroid distances:
    - Neuroticism shift increased to 0.55 because streak_recovery_rate
      needs to reach 0.83 from population mean 0.55 — a large gap
    - Low conscientiousness shift increased to 0.50 because
      completion_ratio_daily needs to drop from 0.50 to 0.22
    - High conscientiousness kept at 0.40 — centroid distance is smaller
    """
    v = vector.copy()

    # ── High conscientiousness → Consistent Achiever ──────────────
    # Centroid distances: consistency +0.21, completion_daily +0.25,
    # motivation_slope +1.26, task_creation +1.76
    if c > 0.5:
        shift = (c - 0.5) * 0.40 * 2          # max shift = 0.40
        v["consistency_score"]      += shift * (0.7120 - POPULATION_DEFAULTS["consistency_score"])
        v["completion_ratio_daily"] += shift * (0.7510 - POPULATION_DEFAULTS["completion_ratio_daily"])
        v["motivation_curve_slope"] += shift * (1.2821 - POPULATION_DEFAULTS["motivation_curve_slope"])
        v["task_creation_rate"]     += shift * (5.2573 - POPULATION_DEFAULTS["task_creation_rate"])
        v["streak_recovery_rate"]   -= shift * (POPULATION_DEFAULTS["streak_recovery_rate"] - 0.5526)
        v["avg_session_hour"]       -= shift * (POPULATION_DEFAULTS["avg_session_hour"] - 12.6782)

    # ── High neuroticism → Burnout-Prone ──────────────────────────
    # Centroid distances: streak_recovery +0.28, consistency -0.29,
    # motivation_slope -1.36 — large gaps need larger shift
    if n > 0.5:
        shift = (n - 0.5) * 0.55 * 2          # max shift = 0.55
        v["streak_recovery_rate"]   += shift * (0.8303 - POPULATION_DEFAULTS["streak_recovery_rate"])
        v["consistency_score"]      -= shift * (POPULATION_DEFAULTS["consistency_score"] - 0.2095)
        v["motivation_curve_slope"] -= shift * (POPULATION_DEFAULTS["motivation_curve_slope"] - (-1.3393))
        v["onboarding_n"]            = n       # direct inject — mirrors streak_recovery

    # ── Low conscientiousness → Easily Distracted ─────────────────
    # Centroid distances: completion_daily -0.28, burst +0.06,
    # task_creation -1.94 — completion gap is large
    if c < 0.5:
        shift = (0.5 - c) * 0.50 * 2          # max shift = 0.50
        v["completion_ratio_daily"] -= shift * (POPULATION_DEFAULTS["completion_ratio_daily"] - 0.2229)
        v["points_burst_ratio"]     += shift * (0.5634 - POPULATION_DEFAULTS["points_burst_ratio"])
        v["task_creation_rate"]     -= shift * (POPULATION_DEFAULTS["task_creation_rate"] - 1.5601)
        v["onboarding_c"]            = c       # direct inject

    # ── Clip all values to valid ranges ───────────────────────────
    v["consistency_score"]      = max(0.0, min(1.0, v["consistency_score"]))
    v["streak_recovery_rate"]   = max(0.0, min(1.0, v["streak_recovery_rate"]))
    v["completion_ratio_daily"] = max(0.0, min(1.0, v["completion_ratio_daily"]))
    v["points_burst_ratio"]     = max(0.0, min(1.0, v["points_burst_ratio"]))
    v["task_creation_rate"]     = max(0.0, min(7.0, v["task_creation_rate"]))
    v["motivation_curve_slope"] = max(-5.0, min(5.0, v["motivation_curve_slope"]))
    v["avg_session_hour"]       = max(7.0, min(23.0, v["avg_session_hour"]))

    return v


# ─── MASTER COLD START BUILDER ────────────────────────────────────────────────

def build_cold_start_vector(
    onboarding_c: float = 0.5,
    onboarding_n: float = 0.5,
    onboarding_o: float = 0.5,
    partial_features: dict = None,
) -> dict:
    """
    Returns a feature dict for cold start prediction.

    Strategy:
    1. Start with real population means (not zeros)
    2. Apply onboarding personality shift (tilts toward likely archetype)
    3. Override with any real features already computed
       (e.g. user has 3 days of data — use those real values)

    This means:
    - Day 0 with onboarding  → population mean + personality tilt
    - Day 0 without onboard  → pure population mean
    - Day 1-13               → mix of real + population for missing features
    - Day 14+                → fully real (cold start not used)
    """
    # Step 1: Start with real population means
    vector = POPULATION_DEFAULTS.copy()

    # Step 2: Inject onboarding scores
    vector["onboarding_c"] = onboarding_c
    vector["onboarding_n"] = onboarding_n
    vector["onboarding_o"] = onboarding_o

    # Step 3: Apply personality-based centroid shift
    vector = _apply_onboarding_shift(vector, onboarding_c, onboarding_n, onboarding_o)

    # Step 4: Override with any real data already available
    # Only override if the real value is meaningfully different
    # from zero (which would indicate missing data, not real zero)
    ZERO_THRESHOLD = {
        "focus_sessions_per_day": 0.1,   # must have at least 0.1 sessions
        "task_creation_rate":     0.1,   # must have created at least 1 task
        "consistency_score":      0.01,  # any non-zero streak counts
        "motivation_curve_slope": None,  # always override (can legitimately be 0)
        "points_burst_ratio":     None,  # always override
        "weekly_point_delta":     None,  # always override
    }

    if partial_features:
        for key, value in partial_features.items():
            if key.startswith("_") or key not in vector:
                continue

            threshold = ZERO_THRESHOLD.get(key)
            if threshold is not None:
                # Only use real value if it exceeds minimum threshold
                if abs(value) >= threshold:
                    vector[key] = value
            else:
                # No threshold — always use real value
                vector[key] = value

    return vector
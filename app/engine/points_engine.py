"""
app/engine/points_engine.py

Computes ML-driven dynamic point multipliers per user archetype.

Research contribution: points become personally meaningful rather
than arbitrary fixed values. Multipliers reward users for completing
tasks that are genuinely difficult given their behavioral archetype.

Formula:
  final_points = base_points × archetype_multiplier × difficulty_multiplier

Archetype multipliers derived from inverse of archetype centroid
completion ratios — archetypes with lower natural completion rates
get higher multipliers for completing those task types.
"""

# ─── ARCHETYPE × TASK TYPE MULTIPLIERS ───────────────────────────────────────
# Rows = archetype, Cols = task interval
# Values derived from inverse completion ratio centroids:
# Easily Distracted daily completion = 0.22 → multiplier = 1/0.22 ≈ 1.5 (capped)
# Consistent Achiever daily = 0.75 → multiplier = 1.0 (baseline)

POINT_MULTIPLIERS = {
    "Consistent Achiever": {
        "Daily":   1.0,   # baseline — easy for them
        "Weekly":  1.1,
        "Monthly": 1.2,
    },
    "Night Owl Performer": {
        "Daily":   1.2,   # harder — inconsistent daytime
        "Weekly":  1.0,
        "Monthly": 1.1,
    },
    "Easily Distracted User": {
        "Daily":   1.5,   # hardest — completion ratio 0.22
        "Weekly":  1.2,
        "Monthly": 1.0,
    },
    "Last-Minute Performer": {
        "Daily":   1.3,
        "Weekly":  1.0,
        "Monthly": 1.0,   # natural strength
    },
    "Burnout-Prone User": {
        "Daily":   1.2,
        "Weekly":  1.3,
        "Monthly": 1.5,   # hardest — long horizon is dangerous for them
    },
}

# Motivation curve slope bonus
# Declining motivation → extra points to re-engage
# Rising motivation → standard points (already motivated)
def _motivation_bonus(motivation_slope: float) -> float:
    if motivation_slope <= -2.0:
        return 1.3    # strong decline → 30% bonus to re-engage
    elif motivation_slope <= -0.5:
        return 1.15   # mild decline → 15% bonus
    elif motivation_slope >= 1.5:
        return 0.9    # rising fast → slight reduction (already engaged)
    return 1.0        # stable → no adjustment


def compute_point_multiplier(
    archetype: str,
    task_interval: str,
    motivation_slope: float = 0.0,
    days_active: int = 0,
) -> float:
    """
    Compute the final point multiplier for a task completion.

    Args:
        archetype:        user's current archetype label
        task_interval:    "Daily", "Weekly", or "Monthly"
        motivation_slope: from feature_snapshot (negative = declining)
        days_active:      used to scale down multiplier for new users
                          (cold start users get neutral multiplier)

    Returns:
        float multiplier (1.0 = standard, >1.0 = bonus, <1.0 = reduced)
    """
    # Cold start — not enough data to trust archetype fully
    # Use neutral multiplier
    if days_active < 3:
        return 1.0

    # Get base archetype multiplier
    archetype_mult = POINT_MULTIPLIERS.get(
        archetype,
        POINT_MULTIPLIERS["Consistent Achiever"]  # fallback
    ).get(task_interval, 1.0)

    # Apply motivation bonus
    motivation_mult = _motivation_bonus(motivation_slope)

    # Blend: archetype weight decays as user matures
    # New users (day 3-14): mostly archetype multiplier
    # Veteran users (day 60+): equal blend
    import math
    personal_weight = min(0.5, days_active / 120.0)
    archetype_weight = 1.0 - personal_weight

    blended = (archetype_weight * archetype_mult +
               personal_weight * motivation_mult)

    # Cap between 0.8 and 1.6 — never penalize more than 20%
    # never reward more than 60%
    return round(max(0.8, min(1.6, blended)), 3)


def apply_multiplier(base_points: int, multiplier: float) -> int:
    """Apply multiplier and return integer points"""
    return max(1, round(base_points * multiplier))
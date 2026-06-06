import math
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np

from app.supabase_client import (
    fetch_user,
    fetch_tasks,
    fetch_work_sessions,
    fetch_points_log,
    fetch_ml_profile,
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _parse_dt(value: str | None) -> Optional[datetime]:
    """Safely parse ISO8601 string → datetime with UTC timezone"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _days_active(sessions: list, points_log: list) -> int:
    """
    Count how many distinct calendar days this user has ANY activity.
    Uses both WorkSessions and PointsLog as activity signals.
    """
    days = set()
    for s in sessions:
        dt = _parse_dt(s.get("timestamp"))
        if dt:
            days.add(dt.date())
    for p in points_log:
        dt = _parse_dt(p.get("created_at"))
        if dt:
            days.add(dt.date())
    return len(days)


def _rolling_window(items: list, field: str, days: int) -> list:
    """Filter a list of records to only those within the last N days"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = []
    for item in items:
        dt = _parse_dt(item.get(field))
        if dt and dt >= cutoff:
            result.append(item)
    return result


# ─── INDIVIDUAL FEATURE COMPUTERS ────────────────────────────────────────────

def compute_completion_ratios(tasks: list) -> dict:
    """
    completion_ratio_daily / weekly / monthly
    = done tasks / total tasks per interval type
    Returns 0.5 (neutral) if no tasks exist for that interval.
    """
    ratios = {}
    for interval in ["Daily", "Weekly", "Monthly"]:
        scoped = [t for t in tasks if t.get("choose") == interval]
        if not scoped:
            ratios[f"completion_ratio_{interval.lower()}"] = 0.5
        else:
            done = sum(1 for t in scoped if t.get("done") is True)
            ratios[f"completion_ratio_{interval.lower()}"] = round(done / len(scoped), 4)
    return ratios


def compute_focus_features(sessions: list) -> dict:
    """
    focus_sessions_per_day : rolling 14d avg sessions per active day
    avg_session_hour        : mean hour-of-day across all sessions
    session_hour_variance   : std dev of session hours (schedule consistency)
    """
    recent = _rolling_window(sessions, "timestamp", 14)

    # Sessions per day (only count days that had at least 1 session)
    day_counts: dict = {}
    hours = []
    for s in recent:
        dt = _parse_dt(s.get("timestamp"))
        if dt:
            day_key = dt.date()
            day_counts[day_key] = day_counts.get(day_key, 0) + 1
            hours.append(dt.hour + dt.minute / 60.0)

    if day_counts:
        focus_per_day = round(sum(day_counts.values()) / len(day_counts), 4)
    else:
        focus_per_day = 0.0

    if hours:
        avg_hour = round(float(np.mean(hours)), 4)
        hour_variance = round(float(np.std(hours)), 4)
    else:
        # Population defaults from student dataset
        avg_hour = 15.3
        hour_variance = 3.5

    return {
        "focus_sessions_per_day": focus_per_day,
        "avg_session_hour": avg_hour,
        "session_hour_variance": hour_variance,
    }


def compute_consistency_score(user: dict) -> dict:
    """
    consistency_score : currentstreak normalized over 30 days [0, 1]
    streak_recovery_rate will be computed separately via reset log
    but we approximate it here from streak vs days_active ratio
    """
    streak = user.get("currentstreak", 0) or 0
    # Normalize: 30-day streak = perfect score of 1.0
    consistency = round(min(streak / 30.0, 1.0), 4)
    return {"consistency_score": consistency, "raw_streak": streak}


def compute_motivation_curve(points_log: list) -> dict:
    """
    motivation_curve_slope : linear regression slope of cumulative
                             daily points over last 14 days.
                             Positive = rising motivation
                             Negative = declining motivation
                             Near zero = stable

    points_burst_ratio     : std dev / mean of daily points
                             High = burst worker, Low = steady worker
    """
    recent = _rolling_window(points_log, "created_at", 14)

    # Aggregate points by day
    daily_points: dict = {}
    for p in recent:
        dt = _parse_dt(p.get("created_at"))
        if dt:
            key = dt.date()
            daily_points[key] = daily_points.get(key, 0) + (p.get("points") or 0)

    if len(daily_points) < 2:
        return {
            "motivation_curve_slope": 0.0,
            "points_burst_ratio": 0.0,
        }

    sorted_days = sorted(daily_points.keys())
    # x = day index (0, 1, 2...), y = points on that day
    x = np.array(range(len(sorted_days)), dtype=float)
    y = np.array([daily_points[d] for d in sorted_days], dtype=float)

    # Linear regression slope via numpy polyfit
    slope = float(np.polyfit(x, y, 1)[0])

    mean_pts = float(np.mean(y))
    std_pts = float(np.std(y))
    burst_ratio = round(std_pts / mean_pts, 4) if mean_pts > 0 else 0.0

    return {
        "motivation_curve_slope": round(slope, 4),
        "points_burst_ratio": burst_ratio,
    }


def compute_weekly_point_delta(points_log: list) -> dict:
    """
    weekly_point_delta : points earned this week minus points earned last week
    Positive = improving, Negative = declining
    """
    now = datetime.now(timezone.utc)

    this_week_start = now - timedelta(days=7)
    last_week_start = now - timedelta(days=14)

    this_week_pts = sum(
        p.get("points", 0) or 0
        for p in points_log
        if _parse_dt(p.get("created_at")) and
           _parse_dt(p.get("created_at")) >= this_week_start
    )
    last_week_pts = sum(
        p.get("points", 0) or 0
        for p in points_log
        if _parse_dt(p.get("created_at")) and
           last_week_start <= _parse_dt(p.get("created_at")) < this_week_start
    )

    delta = this_week_pts - last_week_pts
    return {"weekly_point_delta": float(delta)}


def compute_task_creation_rate(tasks: list) -> dict:
    """
    task_creation_rate : new tasks created in the last 7 days
    Uses Tasks.created_at (the column we added in Step 2)
    """
    recent = _rolling_window(tasks, "created_at", 7)
    return {"task_creation_rate": float(len(recent))}


def compute_streak_recovery_rate(user: dict, days_active: int) -> dict:
    """
    Approximates how many times the user has 'lost' their streak.
    We derive this from: if streak < days_active significantly,
    the user has reset multiple times.
    Formula: resets_approx = max(0, floor(days_active / max(streak,1)) - 1)
    Normalized over 30 days.
    """
    streak = max(user.get("currentstreak", 1) or 1, 1)
    if days_active <= 0:
        return {"streak_recovery_rate": 0.0}

    resets_approx = max(0, math.floor(days_active / streak) - 1)
    # Normalize over 30 days
    rate = round(min(resets_approx / 30.0, 1.0), 4)
    return {"streak_recovery_rate": rate}


# ─── MAIN EXTRACTOR ──────────────────────────────────────────────────────────

def extract_features(uid: str) -> dict:
    """
    Master function. Pulls all Supabase data for a user
    and returns a single flat feature dict ready for the ML model.

    Also returns metadata: days_active, has_enough_data (bool)
    """
    # 1. Fetch all raw data
    user = fetch_user(uid)
    if not user:
        raise ValueError(f"User {uid} not found in Supabase")

    tasks = fetch_tasks(uid)
    sessions = fetch_work_sessions(uid)
    points_log = fetch_points_log(uid)
    ml_profile = fetch_ml_profile(uid)

    # 2. Compute days active
    days = _days_active(sessions, points_log)

    # 3. Compute all feature groups
    features = {}
    features.update(compute_completion_ratios(tasks))
    features.update(compute_focus_features(sessions))
    features.update(compute_consistency_score(user))
    features.update(compute_motivation_curve(points_log))
    features.update(compute_weekly_point_delta(points_log))
    features.update(compute_task_creation_rate(tasks))
    features.update(compute_streak_recovery_rate(user, days))

    # 4. Pull onboarding scores from ml_profile if they exist
    if ml_profile:
        features["onboarding_c"] = ml_profile.get("onboarding_c", 0.5)
        features["onboarding_n"] = ml_profile.get("onboarding_n", 0.5)
        features["onboarding_o"] = ml_profile.get("onboarding_o", 0.5)
    else:
        features["onboarding_c"] = 0.5
        features["onboarding_n"] = 0.5
        features["onboarding_o"] = 0.5

    # 5. Metadata (not fed to model, used by orchestration layer)
    features["_days_active"] = days
    features["_has_enough_data"] = days >= 14
    features["_uid"] = uid

    return features


# ─── ORDERED FEATURE VECTOR (what the model actually sees) ───────────────────

# This exact order must match training data columns — never change order
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


def features_to_vector(features: dict) -> list:
    """
    Convert feature dict → ordered list matching FEATURE_COLUMNS.
    This is what gets passed to model.predict()
    """
    return [features.get(col, 0.0) for col in FEATURE_COLUMNS]
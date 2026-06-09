"""
app/engine/suggestion_engine.py

Suggests what type of task to create next based on
archetype + current behavioral state.
"""

from typing import Optional


SUGGESTIONS = {
    "Consistent Achiever": {
        "rising":   {
            "interval":    "Monthly",
            "message":     "You're on a roll — set a big monthly goal!",
            "reasoning":   "Rising motivation + consistent archetype = ideal time for long-horizon goals",
        },
        "stable":   {
            "interval":    "Daily",
            "message":     "Keep your streak alive with a daily task.",
            "reasoning":   "Stable state — reinforce the daily habit loop",
        },
        "declining": {
            "interval":    "Daily",
            "message":     "Start small today — one daily win.",
            "reasoning":   "Declining motivation — reduce friction with short tasks",
        },
    },
    "Night Owl Performer": {
        "rising":   {
            "interval":    "Weekly",
            "message":     "Great momentum — plan a weekly project.",
            "reasoning":   "Weekly chunks match night owl batch work style",
        },
        "stable":   {
            "interval":    "Weekly",
            "message":     "A weekly goal suits your work rhythm.",
            "reasoning":   "Stable night owl — weekly is natural cadence",
        },
        "declining": {
            "interval":    "Daily",
            "message":     "One small task today to get back on track.",
            "reasoning":   "Declining — reduce scope to daily for re-entry",
        },
    },
    "Easily Distracted User": {
        "rising":   {
            "interval":    "Daily",
            "message":     "Great focus lately — keep daily wins going!",
            "reasoning":   "Even rising, distracted users need short cycles",
        },
        "stable":   {
            "interval":    "Daily",
            "message":     "A quick daily task — just 15 minutes!",
            "reasoning":   "Daily only — long horizons overwhelm",
        },
        "declining": {
            "interval":    "Daily",
            "message":     "One tiny task. Just one.",
            "reasoning":   "Declining distracted user — minimal friction",
        },
    },
    "Last-Minute Performer": {
        "rising":   {
            "interval":    "Monthly",
            "message":     "Motivation is up — set a big deadline goal!",
            "reasoning":   "Rising last-minute performer thrives with deadlines",
        },
        "stable":   {
            "interval":    "Weekly",
            "message":     "A weekly deadline keeps you sharp.",
            "reasoning":   "Weekly deadline suits last-minute work style",
        },
        "declining": {
            "interval":    "Weekly",
            "message":     "Set a deadline for this week — urgency helps you.",
            "reasoning":   "Declining — urgency re-engages last-minute performers",
        },
    },
    "Burnout-Prone User": {
        "rising":   {
            "interval":    "Daily",
            "message":     "Feeling better — a gentle daily task.",
            "reasoning":   "Even rising burnout users need protected scope",
        },
        "stable":   {
            "interval":    "Daily",
            "message":     "Keep it light today — one daily task.",
            "reasoning":   "No long-horizon pressure for burnout users",
        },
        "declining": {
            "interval":    "Daily",
            "message":     "Just one small thing today. Rest is productive too.",
            "reasoning":   "Declining burnout — absolute minimum to prevent collapse",
        },
    },
}


def _classify_slope(motivation_slope: float) -> str:
    if motivation_slope >= 0.8:
        return "rising"
    elif motivation_slope <= -0.5:
        return "declining"
    return "stable"


def get_task_suggestion(
    archetype: str,
    motivation_slope: float = 0.0,
    days_active: int = 0,
) -> dict:
    """
    Returns a task suggestion dict with interval and message.
    """
    if days_active < 1:
        return {
            "interval":  "Daily",
            "message":   "Start with a simple daily task to build your habit.",
            "reasoning": "Cold start — recommend easiest entry point",
            "slope_state": "stable",
        }

    archetype_suggestions = SUGGESTIONS.get(
        archetype,
        SUGGESTIONS["Consistent Achiever"]
    )

    slope_state = _classify_slope(motivation_slope)
    suggestion = archetype_suggestions[slope_state]

    return {
        "interval":    suggestion["interval"],
        "message":     suggestion["message"],
        "reasoning":   suggestion["reasoning"],
        "slope_state": slope_state,
    }
"""
app/engine/personalization.py

Translates an archetype label into concrete app behavior changes.
Rule-based layer that sits on top of the ML prediction.
Returns a recommendations dict that Flutter reads from user_ml_profile.
"""


# ─── POMODORO SETTINGS ────────────────────────────────────────────────────────

POMODORO_SETTINGS = {
    "Consistent Achiever": {
        "work_min":        25,
        "short_break_min": 5,
        "long_break_min":  20,
        "reasoning":       "Standard Pomodoro — you sustain focus well",
    },
    "Night Owl Performer": {
        "work_min":        30,
        "short_break_min": 5,
        "long_break_min":  15,
        "reasoning":       "Longer blocks match your deep work rhythm",
    },
    "Easily Distracted User": {
        "work_min":        15,
        "short_break_min": 5,
        "long_break_min":  10,
        "reasoning":       "Shorter cycles reduce dropout risk",
    },
    "Last-Minute Performer": {
        "work_min":        20,
        "short_break_min": 3,
        "long_break_min":  10,
        "reasoning":       "Urgency-matching rhythm with minimal breaks",
    },
    "Burnout-Prone User": {
        "work_min":        20,
        "short_break_min": 10,
        "long_break_min":  25,
        "reasoning":       "Enforced recovery to prevent collapse",
    },
}


# ─── TASK TYPE WEIGHTS ────────────────────────────────────────────────────────

TASK_WEIGHTS = {
    "Consistent Achiever": {
        "Daily":   0.60,
        "Weekly":  0.30,
        "Monthly": 0.10,
        "reasoning": "Daily rhythm is your strength — reinforce it",
    },
    "Night Owl Performer": {
        "Daily":   0.40,
        "Weekly":  0.50,
        "Monthly": 0.10,
        "reasoning": "Weekly chunks match your batch working style",
    },
    "Easily Distracted User": {
        "Daily":   0.80,
        "Weekly":  0.20,
        "Monthly": 0.00,
        "reasoning": "Small daily wins only — long horizons overwhelm",
    },
    "Last-Minute Performer": {
        "Daily":   0.20,
        "Weekly":  0.50,
        "Monthly": 0.30,
        "reasoning": "Deadline-driven — weekly and monthly goals suit you",
    },
    "Burnout-Prone User": {
        "Daily":   0.50,
        "Weekly":  0.50,
        "Monthly": 0.00,
        "reasoning": "No long-horizon pressure — protect your energy",
    },
}


# ─── NOTIFICATION TIMING ──────────────────────────────────────────────────────

NOTIFICATION_OFFSETS = {
    # Hours relative to user's avg_session_hour
    "Consistent Achiever":    -1,   # 1h early reminder
    "Night Owl Performer":     0,   # exactly at their usual time
    "Easily Distracted User": -2,   # early, before distraction window opens
    "Last-Minute Performer":  +1,   # push during their natural burst window
    "Burnout-Prone User":     -3,   # early, low-pressure, no rush feeling
}

NOTIFICATION_MESSAGES = {
    "Consistent Achiever":    "Time to keep your streak alive! 🔥",
    "Night Owl Performer":    "Your focus window is opening 🌙",
    "Easily Distracted User": "Quick win time — just 15 minutes! ⚡",
    "Last-Minute Performer":  "Crunch time — let's get it done 💪",
    "Burnout-Prone User":     "Gentle start today — you've got this 🌱",
}


# ─── STREAK BEHAVIOR ──────────────────────────────────────────────────────────

STREAK_RULES = {
    "Consistent Achiever": {
        "loss_message":     "comeback",
        "suppress_notifs":  False,
        "reduce_tasks":     False,
        "rest_mode":        False,
    },
    "Night Owl Performer": {
        "loss_message":     "chronotype_shift",
        "suppress_notifs":  False,
        "reduce_tasks":     False,
        "rest_mode":        False,
        "shift_notif_later": True,    # push notification even later
    },
    "Easily Distracted User": {
        "loss_message":     "soft_reentry",
        "suppress_notifs":  False,
        "reduce_tasks":     True,     # drop daily task count by 1 for 3 days
        "reduce_tasks_days": 3,
        "rest_mode":        False,
    },
    "Last-Minute Performer": {
        "loss_message":     "fresh_start",  # no penalty framing
        "suppress_notifs":  False,
        "reduce_tasks":     False,
        "rest_mode":        False,
    },
    "Burnout-Prone User": {
        "loss_message":     "rest_encouraged",
        "suppress_notifs":  True,     # suppress for 24h
        "reduce_tasks":     False,
        "rest_mode":        True,     # flag rest_mode in user_ml_profile
    },
}


# ─── MASTER PERSONALIZATION FUNCTION ─────────────────────────────────────────

def get_recommendations(
    archetype: str,
    avg_session_hour: float = 14.5,
) -> dict:
    """
    Given an archetype label and the user's avg session hour,
    return the full recommendations payload.

    This dict gets stored in user_ml_profile.feature_snapshot
    and returned by the /predict endpoint for Flutter to consume.
    """

    archetype = archetype if archetype in POMODORO_SETTINGS else "Consistent Achiever"

    # Notification hour — clamp to valid range [7, 23]
    offset = NOTIFICATION_OFFSETS[archetype]
    notif_hour = int(max(7, min(23, avg_session_hour + offset)))

    return {
        "pomodoro":      POMODORO_SETTINGS[archetype],
        "task_weights":  TASK_WEIGHTS[archetype],
        "notification": {
            "hour":    notif_hour,
            "message": NOTIFICATION_MESSAGES[archetype],
        },
        "streak_rules":  STREAK_RULES[archetype],
    }
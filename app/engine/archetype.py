"""
app/engine/archetype.py

Core prediction engine. Handles:
- Global model prediction (cold start + all users)
- Personal model training + prediction (14+ days users)
- Alpha-decay ensemble blending
- Archetype assignment with confidence gating
- Persistent personal models (survives restarts)
"""

import math
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import GaussianNB

from app.features.extractor import (
    extract_features,
    features_to_vector,
    FEATURE_COLUMNS,
)
from app.features.cold_start import build_cold_start_vector
from app.supabase_client import (
    fetch_ml_profile,
    upsert_ml_profile,
    log_prediction,
    log_archetype_transition,
)

# ─── LOAD GLOBAL MODEL ONCE AT STARTUP ───────────────────────────────────────
MODEL_DIR    = Path(__file__).parent.parent / "models"
MODEL_PATH   = MODEL_DIR / "global_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

global_model  = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

ARCHETYPE_CLASSES = list(label_encoder.classes_)

# ─── PERSONAL MODEL STORE ────────────────────────────────────────────────────
# Persisted to disk so models survive service restarts.
# In-memory cache for fast access during runtime.

PERSONAL_MODELS_DIR = MODEL_DIR / "personal"
PERSONAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_personal_models: dict[str, GaussianNB] = {}


def _personal_model_path(uid: str) -> Path:
    """Disk path for a user's personal GNB model"""
    return PERSONAL_MODELS_DIR / f"{uid}.pkl"


def _load_personal_model(uid: str) -> GaussianNB | None:
    """
    Attempt to load a personal model from disk into memory cache.
    Returns the model if found, None otherwise.
    """
    path = _personal_model_path(uid)
    if path.exists():
        try:
            model = joblib.load(path)
            _personal_models[uid] = model
            return model
        except Exception:
            # Corrupted file — treat as no model
            return None
    return None


# ─── ALPHA DECAY ─────────────────────────────────────────────────────────────

def compute_alpha(days_active: int) -> float:
    """
    Alpha controls how much we trust the global (dataset-primed) model
    vs the personal (user-specific) model.

    α = 1.0  → fully global (cold start, day 0)
    α = 0.2  → floor — global always contributes at least 20%

    Formula: α = max(0.2, e^(-0.03 * days_active))

    Transition table:
    Day 0  → α = 1.00  (fully global)
    Day 7  → α = 0.81
    Day 14 → α = 0.66
    Day 23 → α = 0.50  (equal blend)
    Day 30 → α = 0.41
    Day 54 → α = 0.20  (floor reached)
    Day 60+ → α = 0.20 (stays at floor)

    Floor raised from 0.1 → 0.2 so global model always
    contributes at least 20%, preventing personal model
    from fully overriding population prior.
    This is more defensible in a research context because
    it keeps the system robust against unusual behavior streaks.
    """
    return round(max(0.2, math.exp(-0.03 * days_active)), 4)


# ─── GLOBAL MODEL PREDICTION ─────────────────────────────────────────────────

def predict_global(feature_vector: list) -> dict:
    """
    Run the global Random Forest on a feature vector.
    Returns probability distribution across all 5 archetypes.
    """
    X = np.array(feature_vector).reshape(1, -1)
    proba = global_model.predict_proba(X)[0]

    return {
        archetype: round(float(prob), 4)
        for archetype, prob in zip(ARCHETYPE_CLASSES, proba)
    }


# ─── PERSONAL MODEL ──────────────────────────────────────────────────────────

def train_personal_model(
    uid: str,
    history_vectors: list,
    history_labels: list,
) -> None:
    """
    Train a Gaussian Naive Bayes model on a user's personal
    feature history and persist it to disk immediately.

    history_vectors : list of feature vectors (each a list of 15 floats)
    history_labels  : list of archetype strings matching each vector

    Requires at least 3 data points to train.
    Persists to disk so model survives service restarts.
    """
    if len(history_vectors) < 3:
        return

    X = np.array(history_vectors)

    try:
        y = label_encoder.transform(history_labels)
    except ValueError as e:
        # history_labels contains an unknown archetype string
        # This should never happen but guard against it
        raise ValueError(f"Unknown archetype label in history: {e}")

    gnb = GaussianNB()
    gnb.fit(X, y)

    # Save to in-memory cache
    _personal_models[uid] = gnb

    # Persist to disk — survives restarts
    joblib.dump(gnb, _personal_model_path(uid))


def predict_personal(uid: str, feature_vector: list) -> dict | None:
    """
    Run the personal GNB model if it exists for this user.

    Fix 1: Checks disk if not in memory cache (survives restarts).
    Fix 2: Safely handles case where user history has fewer than
            5 classes — initializes all to 0.0 then fills known classes,
            then renormalizes so probabilities always sum to 1.0.

    Returns probability dict or None if no personal model exists.
    """
    # Check memory cache first, then try loading from disk
    if uid not in _personal_models:
        loaded = _load_personal_model(uid)
        if loaded is None:
            return None

    gnb = _personal_models[uid]
    X = np.array(feature_vector).reshape(1, -1)
    proba = gnb.predict_proba(X)[0]

    # Initialize all 5 archetypes to 0.0
    result = {archetype: 0.0 for archetype in ARCHETYPE_CLASSES}

    # gnb.classes_ only contains encoded classes seen during training
    # Safely map each back to its archetype name
    for encoded_class, prob in zip(gnb.classes_, proba):
        archetype_name = ARCHETYPE_CLASSES[encoded_class]
        result[archetype_name] = round(float(prob), 4)

    # Renormalize — unseen classes got 0.0 so total may be < 1.0
    total = sum(result.values())
    if total > 0:
        result = {k: round(v / total, 4) for k, v in result.items()}

    return result


# ─── ENSEMBLE BLEND ──────────────────────────────────────────────────────────

def blend_predictions(
    global_proba: dict,
    personal_proba: dict | None,
    alpha: float,
) -> dict:
    """
    Weighted ensemble blend:
    P_final(archetype) = α × P_global(archetype) + (1-α) × P_personal(archetype)

    If no personal model exists yet, returns global_proba unchanged
    (effectively α = 1.0 regardless of computed alpha).
    """
    if personal_proba is None:
        return global_proba

    blended = {}
    for archetype in ARCHETYPE_CLASSES:
        g = global_proba.get(archetype, 0.0)
        p = personal_proba.get(archetype, 0.0)
        blended[archetype] = round(alpha * g + (1.0 - alpha) * p, 4)

    return blended


# ─── ARCHETYPE SELECTION ─────────────────────────────────────────────────────

def select_archetype(blended_proba: dict) -> tuple[str, float]:
    """
    Pick the archetype with the highest blended probability.
    Returns (archetype_name, confidence_score).
    """
    best       = max(blended_proba, key=blended_proba.get)
    confidence = blended_proba[best]
    return best, confidence


# ─── REASSIGNMENT GATE ───────────────────────────────────────────────────────

def should_reassign(
    current_archetype: str,
    new_archetype: str,
    confidence: float,
    min_confidence: float = 0.65,
) -> bool:
    """
    Only reassign the archetype label if:
    1. The prediction actually changed from current
    2. Confidence meets or exceeds threshold

    Prevents thrashing (rapid back-and-forth between labels)
    when the model is uncertain.
    """
    if current_archetype == new_archetype:
        return False
    return confidence >= min_confidence


# ─── MASTER PREDICT FUNCTION ─────────────────────────────────────────────────

def run_prediction(uid: str, trigger_event: str = "manual") -> dict:
    """
    Full prediction pipeline for one user.

    Steps:
    1.  Extract features from Supabase
    2.  Load existing ML profile
    3.  Build feature vector (cold start vs real)
    4.  Compute alpha decay weight
    5.  Run global Random Forest
    6.  Run personal GNB if available
    7.  Blend with alpha decay formula
    8.  Select best archetype
    9.  Apply commitment gate (provisional flag for early cold start)
    10. Apply reassignment gate (prevents thrashing)
    11. Write result to user_ml_profile + prediction_logs
    12. Return full result dict
    """

    # ── 1. Extract real features from Supabase ────────────────────
    raw_features = extract_features(uid)
    days_active  = raw_features["_days_active"]
    has_enough   = raw_features["_has_enough_data"]  # True if days >= 14

    # ── 2. Load existing ML profile ───────────────────────────────
    ml_profile        = fetch_ml_profile(uid)
    # None means first time — no existing assignment yet
    current_archetype = (ml_profile or {}).get("archetype", None)

    onboarding_c = (ml_profile or {}).get("onboarding_c", 0.5)
    onboarding_n = (ml_profile or {}).get("onboarding_n", 0.5)
    onboarding_o = (ml_profile or {}).get("onboarding_o", 0.5)

    # ── 3. Build feature vector ───────────────────────────────────
    if not has_enough:
        # Cold start path:
        # Fill missing features with population defaults
        # but override with any real data we already have
        cold_vector_dict = build_cold_start_vector(
            onboarding_c=onboarding_c,
            onboarding_n=onboarding_n,
            onboarding_o=onboarding_o,
            partial_features=raw_features,
        )
        feature_vector = features_to_vector(cold_vector_dict)
        data_source = "cold_start" if days_active == 0 else "hybrid_early"
    else:
        # Full real feature vector — no population defaults needed
        feature_vector = features_to_vector(raw_features)
        data_source = "hybrid" if days_active < 60 else "personal"

    # ── 4. Compute alpha ──────────────────────────────────────────
    alpha = compute_alpha(days_active)

    # ── 5. Global model prediction ────────────────────────────────
    global_proba = predict_global(feature_vector)

    # ── 6. Personal model prediction (if available) ───────────────
    personal_proba = predict_personal(uid, feature_vector)

    # ── 7. Blend ──────────────────────────────────────────────────
    blended_proba = blend_predictions(global_proba, personal_proba, alpha)

    # ── 8. Select best archetype ──────────────────────────────────
    new_archetype, confidence = select_archetype(blended_proba)

    # ── 9. Commitment gate ────────────────────────────────────────
    # If very early cold start with low confidence,
    # mark prediction as provisional so Flutter can show
    # "Still learning your patterns..." instead of a hard label.
    # Does NOT block assignment — just flags it.
    is_provisional = False
    if days_active < 3 and confidence < 0.45:
        is_provisional = True
        data_source    = "provisional"

# ── 10. Reassignment gate ─────────────────────────────────────
    if current_archetype is None:
        final_archetype = new_archetype

    elif data_source in ("provisional", "cold_start", "hybrid_early"):
        final_archetype = new_archetype

    elif should_reassign(current_archetype, new_archetype, confidence):
        final_archetype = new_archetype
        # ── Transition detected — log it ──────────────────────
        if current_archetype != new_archetype:
            try:
                log_archetype_transition(
                    uid=uid,
                    from_archetype=current_archetype,
                    to_archetype=new_archetype,
                    confidence=confidence,
                    days_active=days_active,
                )
            except Exception as e:
                print(f"Transition log failed (non-critical): {e}")
    else:
        final_archetype = current_archetype

    # ── 11. Write to Supabase ─────────────────────────────────────
    feature_snapshot = {
        k: v for k, v in raw_features.items()
        if not k.startswith("_")
    }

    upsert_ml_profile(uid, {
        "archetype":        final_archetype,
        "confidence":       round(confidence, 4),
        "alpha":            alpha,
        "data_source":      data_source,
        "days_active":      days_active,
        "feature_snapshot": feature_snapshot,
    })

    log_prediction(
        uid=uid,
        archetype=final_archetype,
        confidence=confidence,
        alpha=alpha,
        trigger_event=trigger_event,
        feature_vector=feature_snapshot,
    )



    # ── 12. Return full result ────────────────────────────────────
    return {
        "uid":              uid,
        "archetype":        final_archetype,
        "confidence":       round(confidence, 4),
        "alpha":            alpha,
        "data_source":      data_source,
        "days_active":      days_active,
        "is_provisional":   is_provisional,
        "global_proba":     global_proba,
        "personal_proba":   personal_proba,
        "blended_proba":    blended_proba,
        "trigger_event":    trigger_event,
        "feature_snapshot": feature_snapshot,
    }
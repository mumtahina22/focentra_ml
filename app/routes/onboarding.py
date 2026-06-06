from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.supabase_client import upsert_ml_profile, fetch_ml_profile
from app.engine.archetype import run_prediction
from app.engine.personalization import get_recommendations

router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


class OnboardingRequest(BaseModel):
    uid: str
    # Likert scale 0-4 answers mapped to [0,1] internally
    # Q1: "I always finish tasks before deadlines"  → conscientiousness
    # Q2: "I often feel overwhelmed by my workload" → neuroticism
    # Q3: "I prefer varied tasks over fixed routines" → openness
    answer_c: int = Field(..., ge=0, le=4, description="Conscientiousness 0-4")
    answer_n: int = Field(..., ge=0, le=4, description="Neuroticism 0-4")
    answer_o: int = Field(..., ge=0, le=4, description="Openness 0-4")


def scale_answer(answer: int) -> float:
    """Scale 0-4 Likert answer to [0,1]"""
    return round(answer / 4.0, 4)


@router.post("/")
def submit_onboarding(req: OnboardingRequest):
    """
    Called ONCE right after user registration completes.
    Stores personality proxy scores then immediately runs
    cold start prediction with those scores as priors.
    """
    try:
        c = scale_answer(req.answer_c)
        n = scale_answer(req.answer_n)
        o = scale_answer(req.answer_o)

        # Write onboarding scores first so run_prediction can read them
        upsert_ml_profile(req.uid, {
            "onboarding_c": c,
            "onboarding_n": n,
            "onboarding_o": o,
            "data_source":  "cold_start",
            "days_active":  0,
        })

        # Run cold start prediction with onboarding priors injected
        result = run_prediction(req.uid, trigger_event="onboarding")
        result["recommendations"] = get_recommendations(
            archetype=result["archetype"],
            avg_session_hour=14.5,
        )

        return {
            "status": "onboarding_complete",
            "onboarding_scores": {"c": c, "n": n, "o": o},
            **result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{uid}")
def get_onboarding_status(uid: str):
    """
    Flutter calls this on boot to check if onboarding
    has been completed for this user.
    """
    profile = fetch_ml_profile(uid)
    if not profile:
        return {"completed": False}
    return {
        "completed":   True,
        "archetype":   profile.get("archetype"),
        "data_source": profile.get("data_source"),
        "days_active": profile.get("days_active"),
        "confidence":  profile.get("confidence"),
    }
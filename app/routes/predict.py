from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from app.engine.archetype import run_prediction
from app.engine.personalization import get_recommendations
from app.engine.points_engine import compute_point_multiplier, apply_multiplier
from app.engine.suggestion_engine import get_task_suggestion
from app.supabase_client import (
    fetch_prediction_history,
    fetch_archetype_transitions,
    fetch_ml_profile,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


class PredictRequest(BaseModel):
    uid: Optional[str] = None
    trigger: str = "manual"
    type: Optional[str] = None
    record: Optional[dict] = None
    old_record: Optional[dict] = None
    schema: Optional[str] = None
    table: Optional[str] = None


class PointsRequest(BaseModel):
    uid: str
    base_points: int
    task_interval: str


@router.post("/")
async def predict(request: Request):
    try:
        body = await request.json()
        uid = None

        if "uid" in body and body["uid"] and body["uid"] != "record.uid":
            uid = body["uid"]
        elif "record" in body and body["record"]:
            uid = body["record"].get("uid")

        if not uid:
            raise HTTPException(
                status_code=422,
                detail="Could not extract uid from request body"
            )

        trigger = body.get("trigger", "session_complete")
        if body.get("type") == "INSERT":
            trigger = "session_complete"

        result = run_prediction(uid, trigger_event=trigger)
        result["recommendations"] = get_recommendations(
            archetype=result["archetype"],
            avg_session_hour=result["feature_snapshot"].get("avg_session_hour", 14.5),
        )
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{uid}")
def get_prediction(uid: str):
    try:
        result = run_prediction(uid, trigger_event="boot")
        result["recommendations"] = get_recommendations(
            archetype=result["archetype"],
            avg_session_hour=result["feature_snapshot"].get("avg_session_hour", 14.5),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/points_weight")
def get_points_weight(req: PointsRequest):
    try:
        profile = fetch_ml_profile(req.uid)
        archetype = profile.get("archetype", "Consistent Achiever") if profile else "Consistent Achiever"
        days_active = profile.get("days_active", 0) if profile else 0
        snapshot = profile.get("feature_snapshot", {}) if profile else {}
        motivation_slope = snapshot.get("motivation_curve_slope", 0.0)

        multiplier = compute_point_multiplier(
            archetype=archetype,
            task_interval=req.task_interval,
            motivation_slope=motivation_slope,
            days_active=days_active,
        )
        adjusted_points = apply_multiplier(req.base_points, multiplier)

        return {
            "base_points":     req.base_points,
            "multiplier":      multiplier,
            "adjusted_points": adjusted_points,
            "archetype":       archetype,
            "task_interval":   req.task_interval,
        }
    except Exception as e:
        return {
            "base_points":     req.base_points,
            "multiplier":      1.0,
            "adjusted_points": req.base_points,
            "archetype":       "unknown",
            "task_interval":   req.task_interval,
        }


@router.get("/suggest/{uid}")
def suggest_task(uid: str):
    try:
        profile = fetch_ml_profile(uid)
        archetype = profile.get("archetype", "Consistent Achiever") if profile else "Consistent Achiever"
        days_active = profile.get("days_active", 0) if profile else 0
        snapshot = profile.get("feature_snapshot", {}) if profile else {}
        motivation_slope = snapshot.get("motivation_curve_slope", 0.0)

        suggestion = get_task_suggestion(
            archetype=archetype,
            motivation_slope=motivation_slope,
            days_active=days_active,
        )
        suggestion["archetype"] = archetype
        return suggestion
    except Exception as e:
        return {
            "interval":    "Daily",
            "message":     "Start with a daily task!",
            "reasoning":   "Fallback suggestion",
            "slope_state": "stable",
            "archetype":   "unknown",
        }


@router.get("/history/{uid}")
def get_prediction_history(uid: str):
    try:
        history = fetch_prediction_history(uid, limit=30)
        transitions = fetch_archetype_transitions(uid, limit=10)
        return {
            "history":     history,
            "transitions": transitions,
            "count":       len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
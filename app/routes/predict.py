from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.engine.archetype import run_prediction
from app.engine.personalization import get_recommendations

router = APIRouter(prefix="/predict", tags=["Prediction"])


class PredictRequest(BaseModel):
    uid: str
    trigger: str = "manual"


@router.post("/")
def predict(req: PredictRequest):
    """
    Main prediction endpoint.
    Called by Supabase webhook on WorkSessions INSERT.
    Also called by Flutter manually on boot.
    """
    try:
        result = run_prediction(req.uid, trigger_event=req.trigger)
        result["recommendations"] = get_recommendations(
            archetype=result["archetype"],
            avg_session_hour=result["feature_snapshot"].get("avg_session_hour", 14.5),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{uid}")
def get_prediction(uid: str):
    """
    Lightweight GET for Flutter boot check.
    Reruns prediction and returns fresh result.
    """
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
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from app.engine.archetype import run_prediction
from app.engine.personalization import get_recommendations

router = APIRouter(prefix="/predict", tags=["Prediction"])


class PredictRequest(BaseModel):
    uid: Optional[str] = None
    trigger: str = "manual"
    # Supabase webhook fields
    type: Optional[str] = None
    record: Optional[dict] = None
    old_record: Optional[dict] = None
    schema: Optional[str] = None
    table: Optional[str] = None


@router.post("/")
async def predict(request: Request):
    """
    Handles both:
    1. Direct Flutter call: {"uid": "xxx", "trigger": "boot"}
    2. Supabase webhook: {"type": "INSERT", "record": {"uid": "xxx", ...}}
    """
    try:
        body = await request.json()
        

        print("🔥 RAW WEBHOOK BODY:", body)

        # Extract uid from either format
        uid = None

        # Format 1: direct call with uid field
        if "uid" in body and body["uid"] and body["uid"] != "record.uid":
            uid = body["uid"]

        # Format 2: Supabase webhook with record object
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
            avg_session_hour=result["feature_snapshot"].get(
                "avg_session_hour", 14.5
            ),
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
            avg_session_hour=result["feature_snapshot"].get(
                "avg_session_hour", 14.5
            ),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

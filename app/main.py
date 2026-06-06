import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from app.routes.predict    import router as predict_router
from app.routes.onboarding import router as onboarding_router
from app.scheduler.retrain_job import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Focentra ML Service starting up...")
    scheduler = start_scheduler()
    logger.info("All systems ready.")
    yield
    scheduler.shutdown()
    logger.info("Service stopping.")


app = FastAPI(
    title="Focentra ML Service",
    description="Cold-start adaptive productivity archetype engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(onboarding_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "focentra-ml", "version": "1.0.0"}


@app.get("/test-db")
def test_db():
    from app.supabase_client import get_client
    try:
        client = get_client()
        result = client.from_("Users").select("id").limit(1).execute()
        return {"status": "connected", "rows": len(result.data)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/test-features/{uid}")
def test_features(uid: str):
    from app.features.extractor import extract_features, features_to_vector
    features = extract_features(uid)
    vector   = features_to_vector(features)
    return {
        "features":        features,
        "vector_length":   len(vector),
        "days_active":     features["_days_active"],
        "has_enough_data": features["_has_enough_data"],
    }


@app.get("/test-predict/{uid}")
def test_predict(uid: str):
    from app.engine.archetype import run_prediction
    from app.engine.personalization import get_recommendations
    result = run_prediction(uid, trigger_event="test")
    result["recommendations"] = get_recommendations(
        archetype=result["archetype"],
        avg_session_hour=result["feature_snapshot"].get("avg_session_hour", 14.5),
    )
    return result
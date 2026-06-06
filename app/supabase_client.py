import os
from datetime import datetime, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

# Loads .env file locally — on Railway env vars are injected directly
# so this is a no-op on production but harmless
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Do NOT raise here at import time — Railway injects vars after module load
# Let the client creation fail naturally if vars are missing
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    import warnings
    warnings.warn(
        "SUPABASE_URL or SUPABASE_SERVICE_KEY not found. "
        "Set these in Railway Variables tab.",
        RuntimeWarning
    )
    # Set placeholder so module loads — will fail at first DB call
    SUPABASE_URL = SUPABASE_URL or "https://placeholder.supabase.co"
    SUPABASE_SERVICE_KEY = SUPABASE_SERVICE_KEY or "placeholder"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_client() -> Client:
    return supabase


# ─── READ HELPERS ────────────────────────────────────────────────────────────

def fetch_user(uid: str) -> dict | None:
    """Fetch a single user row from public.Users"""
    result = supabase.from_("Users").select("*").eq("id", uid).single().execute()
    return result.data


def fetch_tasks(uid: str) -> list:
    """Fetch all tasks for a user"""
    result = supabase.from_("Tasks").select("*").eq("uid", uid).execute()
    return result.data or []


def fetch_points_log(uid: str) -> list:
    """Fetch full points history for a user"""
    result = (
        supabase.from_("PointsLog")
        .select("*")
        .eq("uid", uid)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data or []


def fetch_work_sessions(uid: str) -> list:
    """Fetch all work sessions for a user"""
    result = (
        supabase.from_("WorkSessions")
        .select("*")
        .eq("uid", uid)
        .order("timestamp", desc=False)
        .execute()
    )
    return result.data or []


def fetch_ml_profile(uid: str) -> dict | None:
    """Fetch the ML profile row for a user, or None if not yet created"""
    result = (
        supabase.from_("user_ml_profile")
        .select("*")
        .eq("uid", uid)
        .execute()
    )
    return result.data[0] if result.data else None


def fetch_all_users() -> list:
    """Fetch all user IDs — used by weekly retraining job"""
    result = supabase.from_("Users").select("id").execute()
    return result.data or []


# ─── WRITE HELPERS ────────────────────────────────────────────────────────────

def upsert_ml_profile(uid: str, payload: dict) -> None:
    """
    Create or update the user_ml_profile row.
    Always sets updated_at to now.
    """
    payload["uid"] = uid
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    supabase.from_("user_ml_profile").upsert(payload).execute()


def log_prediction(
    uid: str,
    archetype: str,
    confidence: float,
    alpha: float,
    trigger_event: str,
    feature_vector: dict,
) -> None:
    """Write an immutable prediction record to prediction_logs"""
    supabase.from_("prediction_logs").insert({
        "uid": uid,
        "archetype": archetype,
        "confidence": confidence,
        "alpha": alpha,
        "trigger_event": trigger_event,
        "feature_vector": feature_vector,
    }).execute()
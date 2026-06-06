"""
Weekly retraining job.
Runs every Sunday at 02:00 UTC.
Rebuilds personal GNB models from prediction_logs history.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.supabase_client import get_client, fetch_all_users, fetch_ml_profile
from app.engine.archetype import (
    train_personal_model,
    run_prediction,
    FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def retrain_all_users():
    """
    For every user with 14+ days of data:
    1. Fetch their prediction_logs history (last 90 entries)
    2. Rebuild personal GNB model
    3. Run fresh prediction with new model
    """
    logger.info("Weekly retraining job started")
    client    = get_client()
    users     = fetch_all_users()
    retrained = 0
    skipped   = 0

    for user_row in users:
        uid = user_row["id"]
        try:
            profile = fetch_ml_profile(uid)
            if not profile:
                skipped += 1
                continue

            days_active = profile.get("days_active", 0)
            if days_active < 14:
                skipped += 1
                continue

            # Fetch prediction history for this user
            logs = (
                client
                .from_("prediction_logs")
                .select("feature_vector, archetype")
                .eq("uid", uid)
                .order("created_at", desc=False)
                .limit(90)
                .execute()
            )

            if not logs.data or len(logs.data) < 3:
                skipped += 1
                continue

            # Build training vectors from log history
            vectors = []
            labels  = []
            for log in logs.data:
                fv    = log.get("feature_vector", {})
                label = log.get("archetype")
                if not fv or not label:
                    continue
                vector = [fv.get(col, 0.0) for col in FEATURE_COLUMNS]
                vectors.append(vector)
                labels.append(label)

            if len(vectors) < 3:
                skipped += 1
                continue

            # Retrain and persist personal model
            train_personal_model(uid, vectors, labels)

            # Run fresh prediction using newly trained model
            run_prediction(uid, trigger_event="weekly_retrain")

            retrained += 1
            logger.info(f"Retrained: {uid}")

        except Exception as e:
            logger.error(f"Failed retraining {uid}: {e}")
            continue

    logger.info(
        f"Retraining complete — "
        f"retrained: {retrained}, skipped: {skipped}"
    )


def start_scheduler():
    """
    Start background scheduler on service startup.
    Called once from main.py lifespan.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        retrain_all_users,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_retrain",
        name="Weekly personal model retraining",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started — weekly retraining every Sunday 02:00 UTC")
    return scheduler
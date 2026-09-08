from datetime import datetime, timezone

from pymongo import MongoClient

from config import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

_client = None


def get_collection():
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI is not set (check .env)")
        _client = MongoClient(MONGODB_URI)
    return _client[MONGODB_DB][MONGODB_COLLECTION]


def save_detection(record: dict) -> dict:
    record = {**record, "created_at": datetime.now(timezone.utc)}
    get_collection().insert_one(record)
    record.pop("_id", None)
    return record


def list_detections(folder: str | None = None) -> list[dict]:
    query = {"folder": folder} if folder else {}
    docs = list(get_collection().find(query).sort("created_at", -1))
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
            doc["created_at"] = doc["created_at"].isoformat()
    return docs

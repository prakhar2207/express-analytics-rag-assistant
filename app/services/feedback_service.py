import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.core.config import settings

class FeedbackService:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.FEEDBACK_FILE_PATH)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._save_all([])

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_all(self, items: List[Dict[str, Any]]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    def record_feedback(self, query: str, answer: str, rating: str, comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Record feedback (thumbs up / thumbs down) for a specific answer.
        """
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "rating": rating,
            "comment": comment or ""
        }
        items = self._load_all()
        items.append(entry)
        self._save_all(items)
        return entry

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        return self._load_all()

feedback_service = FeedbackService()

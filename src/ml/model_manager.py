"""Model persistence helpers (serialization + metadata)."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ModelManager:
    """Handles saving/loading ML models from `src/data/models`."""

    def __init__(self, models_dir: Path) -> None:
        """Store the target directory where artifacts will live."""
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, metadata: Dict[str, Any]) -> Path:
        """Persist a trained model plus metadata and return the created path."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_name = metadata.get("name", f"model_{timestamp}")
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}.json"

        with model_path.open("wb") as fh:
            pickle.dump(model, fh)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return model_path

    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a previously saved model along with its metadata."""
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with model_path.open("rb") as fh:
            model = pickle.load(fh)
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        return model, metadata

    def list_available_models(self) -> List[str]:
        """Return identifiers for all persisted models."""
        models = []
        for file in self.models_dir.glob("*.pkl"):
            models.append(file.stem)
        return sorted(models)

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import LOGS_DIR, PROFILES_DIR, SESSIONS_DIR, STORAGE_DIR


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_storage_layout() -> None:
    for folder in [
        STORAGE_DIR,
        LOGS_DIR,
        PROFILES_DIR,
        SESSIONS_DIR / "calibration",
        SESSIONS_DIR / "training",
        SESSIONS_DIR / "inference",
        SESSIONS_DIR / "simulation",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def profile_dir(profile_id: str) -> Path:
    return PROFILES_DIR / profile_id


def profile_file(profile_id: str) -> Path:
    return profile_dir(profile_id) / "profile.json"


def profile_raw_edf_dir(profile_id: str) -> Path:
    return profile_dir(profile_id) / "raw_edf"


def profile_preprocessed_dir(profile_id: str) -> Path:
    return profile_dir(profile_id) / "preprocessed_edf"


def profile_segmented_dir(profile_id: str) -> Path:
    return profile_dir(profile_id) / "segmented_windows"


def profile_models_dir(profile_id: str) -> Path:
    return profile_dir(profile_id) / "models"


def profile_training_dir(profile_id: str) -> Path:
    return profile_dir(profile_id) / "training"


def profile_inference_dir(profile_id: str) -> Path:
    return profile_dir(profile_id) / "inference"


def ensure_profile_layout(profile_id: str) -> None:
    for folder in [
        profile_dir(profile_id),
        profile_raw_edf_dir(profile_id),
        profile_preprocessed_dir(profile_id),
        profile_segmented_dir(profile_id),
        profile_models_dir(profile_id),
        profile_training_dir(profile_id),
        profile_inference_dir(profile_id),
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_existing_edf(paths: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for item in paths:
        candidate = Path(item)
        if candidate.exists() and candidate.suffix.lower() == ".edf":
            resolved.append(candidate)
    return resolved


def list_profile_edf(profile_id: str) -> List[Path]:
    folder = profile_raw_edf_dir(profile_id)
    if not folder.exists():
        return []
    return sorted([path for path in folder.glob("*.edf") if path.is_file()])


def write_session_manifest(session_type: str, session_id: str, payload: Dict[str, Any]) -> Path:
    path = SESSIONS_DIR / session_type / f"{session_id}.json"
    write_json(path, payload)
    return path

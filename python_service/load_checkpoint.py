from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() != ".pt":
        raise ValueError("Checkpoint must be a .pt file.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"weights": payload, "model_type": "diff_e"}


def save_user_checkpoint(base_checkpoint: Dict[str, Any], output_path: str, metadata: Dict[str, Any]) -> str:
    checkpoint = dict(base_checkpoint)
    checkpoint["model_type"] = "diff_e"
    checkpoint["personalization"] = metadata

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return str(path)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import WORD_WINDOWS_SECONDS
from .storage import profile_segmented_dir


@dataclass
class SegmentOutput:
    windows: List[np.ndarray]
    window_paths: List[Path]
    metadata: List[Dict[str, object]]


def _slice_or_pad(data: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if end_idx <= data.shape[1]:
        return data[:, start_idx:end_idx]

    padded = np.zeros((data.shape[0], end_idx - start_idx), dtype=data.dtype)
    available = max(data.shape[1] - start_idx, 0)
    if available > 0:
        padded[:, :available] = data[:, start_idx : start_idx + available]
    return padded


def segment_imagination_windows(profile_id: str, data: np.ndarray, sfreq: int, source_name: str) -> SegmentOutput:
    segmented_dir = profile_segmented_dir(profile_id)
    segmented_dir.mkdir(parents=True, exist_ok=True)

    windows: List[np.ndarray] = []
    window_paths: List[Path] = []
    metadata: List[Dict[str, object]] = []

    for win in WORD_WINDOWS_SECONDS:
        start, end = win["keep"]
        start_idx = int(round(start * sfreq))
        end_idx = int(round(end * sfreq))
        window = _slice_or_pad(data, start_idx, end_idx)

        slot = int(win["slot"])
        output_path = segmented_dir / f"{source_name}_slot{slot}.npy"
        np.save(output_path, window)

        windows.append(window)
        window_paths.append(output_path)
        metadata.append(
            {
                "slot": slot,
                "keep_window_seconds": [start, end],
                "shape": [int(window.shape[0]), int(window.shape[1])],
                "path": str(output_path),
            }
        )

    return SegmentOutput(windows=windows, window_paths=window_paths, metadata=metadata)

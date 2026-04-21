from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

from .config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    CHANNEL_STD_FLAT_THRESHOLD,
    CHANNEL_STD_NOISY_THRESHOLD,
    EMOTIV_CHANNELS,
    NOTCH_HZ,
    SAMPLING_RATE_HZ,
)
from .storage import profile_preprocessed_dir


@dataclass
class PreprocessOutput:
    data: np.ndarray
    sfreq: int
    channels: List[str]
    report: Dict[str, object]
    output_path: Path


def _load_edf_matrix(edf_path: Path) -> Tuple[np.ndarray, float, List[str]]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    picks = [name for name in EMOTIV_CHANNELS if name in raw.ch_names]
    if len(picks) < 14:
        picks = raw.ch_names[:14]

    matrix = raw.copy().pick(picks).get_data()

    if matrix.shape[0] > 14:
        matrix = matrix[:14, :]
        picks = picks[:14]

    if matrix.shape[0] < 14:
        pad_rows = 14 - matrix.shape[0]
        matrix = np.vstack([matrix, np.zeros((pad_rows, matrix.shape[1]))])
        picks = picks + [f"PAD{idx + 1}" for idx in range(pad_rows)]

    sfreq = float(raw.info["sfreq"])
    return matrix, sfreq, picks


def _mean_remove(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=1, keepdims=True)


def _apply_notch(data: np.ndarray, sfreq: float) -> np.ndarray:
    b, a = iirnotch(NOTCH_HZ, 30.0, sfreq)
    return filtfilt(b, a, data, axis=1)


def _apply_bandpass(data: np.ndarray, sfreq: float) -> np.ndarray:
    b, a = butter(4, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ], btype="bandpass", fs=sfreq)
    return filtfilt(b, a, data, axis=1)


def _resample_if_needed(data: np.ndarray, sfreq: float) -> np.ndarray:
    if int(round(sfreq)) == SAMPLING_RATE_HZ:
        return data
    samples = int(round(data.shape[1] * SAMPLING_RATE_HZ / sfreq))
    return resample(data, samples, axis=1)


def _validate_and_interpolate(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    stds = np.std(data, axis=1)
    bad_indices = [
        int(index)
        for index, value in enumerate(stds)
        if value <= CHANNEL_STD_FLAT_THRESHOLD or value >= CHANNEL_STD_NOISY_THRESHOLD
    ]

    report: Dict[str, object] = {
        "bad_channel_indices": bad_indices,
        "interpolation_applied": False,
    }

    if not bad_indices:
        return data, report

    good_indices = [index for index in range(data.shape[0]) if index not in bad_indices]
    if not good_indices:
        return data, report

    repaired = data.copy()
    replacement = np.mean(repaired[good_indices, :], axis=0)
    for index in bad_indices:
        repaired[index, :] = replacement

    report["interpolation_applied"] = True
    return repaired, report


def preprocess_edf(profile_id: str, edf_path: str) -> PreprocessOutput:
    path = Path(edf_path)
    matrix, sfreq, channels = _load_edf_matrix(path)

    matrix = _mean_remove(matrix)
    matrix = _apply_notch(matrix, sfreq)
    matrix = _apply_bandpass(matrix, sfreq)
    matrix = _resample_if_needed(matrix, sfreq)
    matrix, quality_report = _validate_and_interpolate(matrix)

    output_folder = profile_preprocessed_dir(profile_id)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{path.stem}_preprocessed.npy"
    np.save(output_path, matrix)

    report = {
        "source_edf": str(path),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "target_sfreq": SAMPLING_RATE_HZ,
        "channels": channels,
        "quality": quality_report,
    }

    return PreprocessOutput(
        data=matrix,
        sfreq=SAMPLING_RATE_HZ,
        channels=channels,
        report=report,
        output_path=output_path,
    )

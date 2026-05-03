from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample, welch

from .config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDS,
    CHANNEL_STD_FLAT_THRESHOLD,
    CHANNEL_STD_NOISY_THRESHOLD,
    EMOTIV_CHANNELS,
    NOTCH_HZ,
    SAMPLING_RATE_HZ,
)


@dataclass
class ProcessedTrial:
    edf_path: str
    marker_csv_path: str
    channels: List[str]
    sampling_rate: int
    slot_tensors: List[np.ndarray]
    slot_tensor_shapes: List[List[int]]
    warnings: List[str]
    imagine_markers: List[Dict[str, object]]
    subject: Optional[str]
    trial: Optional[str]
    sentence_id: Optional[str]


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
    target_samples = int(round(data.shape[1] * SAMPLING_RATE_HZ / sfreq))
    return resample(data, target_samples, axis=1)


def _validate_and_interpolate(data: np.ndarray, channel_names: List[str]) -> tuple[np.ndarray, List[str]]:
    stds = np.std(data, axis=1)
    bad_indices = [
        index
        for index, value in enumerate(stds.tolist())
        if value <= CHANNEL_STD_FLAT_THRESHOLD or value >= CHANNEL_STD_NOISY_THRESHOLD
    ]
    if not bad_indices:
        return data, []

    good_indices = [index for index in range(data.shape[0]) if index not in bad_indices]
    if not good_indices:
        raise ValueError("All EEG channels failed quality validation.")

    repaired = data.copy()
    replacement = np.mean(repaired[good_indices, :], axis=0)
    for index in bad_indices:
        repaired[index, :] = replacement
    warnings = [f"Interpolated low-quality channel: {channel_names[index]}" for index in bad_indices]
    return repaired, warnings


def _read_marker_csv(marker_path: Path) -> List[Dict[str, str]]:
    with marker_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Marker CSV is missing a header row.")
        normalized_rows: List[Dict[str, str]] = []
        for row in reader:
            normalized = {str(key).strip().lower(): str(value).strip() for key, value in row.items() if key is not None}
            normalized_rows.append(normalized)
    if not normalized_rows:
        raise ValueError("Marker CSV is empty.")
    return normalized_rows


def _resolve_marker_csv(edf_path: Path, marker_csv_path: Optional[str]) -> Path:
    if marker_csv_path:
        candidate = Path(marker_csv_path)
        if candidate.suffix.lower() != ".csv":
            raise ValueError("Marker file must be a .csv file.")
        if not candidate.exists():
            raise FileNotFoundError(f"Marker CSV not found: {candidate}")
        return candidate

    candidates = sorted(edf_path.parent.glob("*intervalMarker*.csv"))
    if not candidates:
        candidates = sorted(edf_path.parent.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Marker CSV is required but was not found next to the EDF file.")
    return candidates[0]


def _parse_file_context(edf_path: Path) -> Dict[str, Optional[str]]:
    stem = edf_path.stem
    subject = None
    sentence_id = None
    trial = None

    for token in edf_path.parent.name.split("_"):
        token_upper = token.upper()
        if token_upper.startswith("S") and token_upper[1:].isdigit():
            subject = token_upper
        elif token_upper.startswith("C") and token_upper[1:].isdigit():
            sentence_id = token_upper
        elif token_upper.startswith("T") and token_upper[1:].isdigit():
            trial = token_upper

    if sentence_id is None:
        for part in stem.split("_"):
            if part.isdigit():
                sentence_id = f"C{part}"
                break

    return {"subject": subject, "sentence_id": sentence_id, "trial": trial}


def _extract_imagine_markers(rows: List[Dict[str, str]], raw_duration_sec: float, sfreq: float) -> List[Dict[str, object]]:
    if not rows:
        raise ValueError("Marker CSV did not contain any rows.")
    max_latency = max(float(row.get("latency", "nan")) for row in rows if row.get("latency"))
    latency_is_samples = max_latency > (raw_duration_sec * 2.0)

    imagine_markers: List[Dict[str, object]] = []
    for row in rows:
        marker_type = str(row.get("type", ""))
        if "phase_imagine" not in marker_type.lower():
            continue
        latency_raw = row.get("latency", "")
        duration_raw = row.get("duration", "")
        if not latency_raw or not duration_raw:
            continue
        latency_value = float(latency_raw)
        duration_value = float(duration_raw)
        onset_seconds = latency_value / sfreq if latency_is_samples else latency_value
        imagine_markers.append(
            {
                "type": marker_type,
                "latency": latency_value,
                "duration": duration_value,
                "onset_seconds": onset_seconds,
            }
        )

    if not imagine_markers:
        raise ValueError("Marker CSV has no phase_Imagine rows.")

    valid = [
        marker
        for marker in imagine_markers
        if float(marker["duration"]) >= 6.0 - 1e-3
    ]
    valid.sort(key=lambda item: float(item["onset_seconds"]))

    if len(valid) != 3:
        raise ValueError(
            f"Marker CSV does not contain exactly 3 usable imagination phases. Found {len(valid)}."
        )
    for index, marker in enumerate(valid, start=1):
        marker["slot"] = index
    return valid


def _build_band_rows(window: np.ndarray, sfreq: int) -> np.ndarray:
    powers = np.zeros((window.shape[0], len(BANDS)), dtype=np.float32)
    for channel_index, channel in enumerate(window):
        freqs, psd = welch(channel, fs=sfreq, nperseg=min(256, len(channel)))
        for band_index, (low_hz, high_hz) in enumerate(BANDS.values()):
            powers[channel_index, band_index] = float(psd[(freqs >= low_hz) & (freqs < high_hz)].sum())
    mean_band_powers = powers.mean(axis=0)
    return np.repeat(mean_band_powers[:, None], window.shape[1], axis=1).astype(np.float32)


def process_trial_edf(edf_path: str, marker_csv_path: Optional[str] = None) -> ProcessedTrial:
    path = Path(edf_path)
    if path.suffix.lower() != ".edf":
        raise ValueError("EEG recording must be an .edf file.")
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    marker_path = _resolve_marker_csv(path, marker_csv_path)
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    missing_channels = [channel for channel in EMOTIV_CHANNELS if channel not in raw.ch_names]
    if missing_channels:
        raise ValueError(f"EDF is missing required EEG channels: {', '.join(missing_channels)}")

    orig_sfreq = float(raw.info["sfreq"])
    matrix = raw.copy().pick(EMOTIV_CHANNELS).get_data()
    matrix = _mean_remove(matrix)
    matrix = _apply_notch(matrix, orig_sfreq)
    matrix = _apply_bandpass(matrix, orig_sfreq)

    warnings: List[str] = []
    if int(round(orig_sfreq)) != SAMPLING_RATE_HZ:
        warnings.append(f"Resampled EEG from {orig_sfreq:.2f} Hz to {SAMPLING_RATE_HZ} Hz.")
    matrix = _resample_if_needed(matrix, orig_sfreq)
    matrix, quality_warnings = _validate_and_interpolate(matrix, EMOTIV_CHANNELS)
    warnings.extend(quality_warnings)

    raw_duration_sec = raw.n_times / orig_sfreq
    marker_rows = _read_marker_csv(marker_path)
    imagine_markers = _extract_imagine_markers(marker_rows, raw_duration_sec, orig_sfreq)

    slot_tensors: List[np.ndarray] = []
    slot_tensor_shapes: List[List[int]] = []
    for marker in imagine_markers:
        onset_seconds = float(marker["onset_seconds"])
        start_idx = int(round((onset_seconds + 0.5) * SAMPLING_RATE_HZ))
        stop_idx = start_idx + int(round(5.5 * SAMPLING_RATE_HZ))
        if start_idx < 0 or stop_idx > matrix.shape[1]:
            raise ValueError("EDF is too short for the requested imagination windows.")

        eeg_window = matrix[:, start_idx:stop_idx].astype(np.float32)
        if eeg_window.shape != (14, 1408):
            raise ValueError(f"Expected a [14, 1408] EEG window, found {eeg_window.shape}")
        band_rows = _build_band_rows(eeg_window, SAMPLING_RATE_HZ)
        slot_tensor = np.vstack([eeg_window, band_rows]).astype(np.float32)
        if slot_tensor.shape != (19, 1408):
            raise ValueError(f"Expected a [19, 1408] Stage 1 tensor, found {slot_tensor.shape}")
        slot_tensors.append(slot_tensor)
        slot_tensor_shapes.append([int(slot_tensor.shape[0]), int(slot_tensor.shape[1])])

    context = _parse_file_context(path)
    return ProcessedTrial(
        edf_path=str(path),
        marker_csv_path=str(marker_path),
        channels=list(EMOTIV_CHANNELS),
        sampling_rate=SAMPLING_RATE_HZ,
        slot_tensors=slot_tensors,
        slot_tensor_shapes=slot_tensor_shapes,
        warnings=warnings,
        imagine_markers=imagine_markers,
        subject=context["subject"],
        trial=context["trial"],
        sentence_id=context["sentence_id"],
    )

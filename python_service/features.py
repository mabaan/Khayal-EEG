from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

from .config import BANDS


def _band_envelope(signal: np.ndarray, sfreq: int, low_hz: float, high_hz: float) -> np.ndarray:
    b, a = butter(3, [low_hz, high_hz], btype="bandpass", fs=sfreq)
    filtered = filtfilt(b, a, signal)
    analytic = hilbert(filtered)
    return np.abs(analytic)


def build_band_channels(window: np.ndarray, sfreq: int) -> Tuple[np.ndarray, dict]:
    mean_signal = np.mean(window, axis=0)
    band_channels = []
    band_powers = {}

    for name, (low_hz, high_hz) in BANDS.items():
        envelope = _band_envelope(mean_signal, sfreq, low_hz, high_hz)
        band_channels.append(envelope)

        freqs, psd = welch(mean_signal, fs=sfreq, nperseg=min(256, mean_signal.shape[0]))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        power = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        band_powers[name] = power

    return np.vstack(band_channels), band_powers


def build_stage1_tensor(window: np.ndarray, sfreq: int) -> Tuple[np.ndarray, dict]:
    band_channels, band_powers = build_band_channels(window, sfreq)
    tensor = np.vstack([window, band_channels])

    channel_mean = np.mean(tensor, axis=1, keepdims=True)
    channel_std = np.std(tensor, axis=1, keepdims=True) + 1e-6
    normalized = (tensor - channel_mean) / channel_std

    return normalized, band_powers

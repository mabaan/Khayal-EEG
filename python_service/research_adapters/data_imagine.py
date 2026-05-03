# -*- coding: utf-8 -*-
"""
EEG Data Preprocessor for Arabic Imagined Speech.

Loads EDF files, extracts features, and creates LOTO (Leave-One-Trial-Out) splits.
Output: [N, 19, T] tensors built from full trials
"""

import os
import json
import pickle as pkl
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.signal as sps

try:
    import mne
    mne.set_log_level("WARNING")
except ImportError:
    mne = None

WIN_SEC = 2.75  # Kept for CLI/backward compatibility; full trials are used below.
DEFAULT_OVERLAP = 0.0  # Unused when full trials are used.


# =============================================================================
# Signal Processing
# =============================================================================

def _read_channels_file(root: str) -> List[str]:
    """Load 14 channel names from channels_14.txt."""
    path = Path(root) / "channels_14.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    channels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(channels) != 14:
        raise ValueError(f"Expected 14 channels, got {len(channels)}")
    return channels


def filter_eeg(eeg: np.ndarray, fs: int = 256) -> np.ndarray:
    """Bandpass (1-40Hz) and notch (50Hz) filter."""
    b_bp, a_bp = sps.butter(4, [1.0, 40.0], btype="bandpass", fs=fs)
    b_n, a_n = sps.iirnotch(w0=50.0, Q=30.0, fs=fs)
    eeg = sps.filtfilt(b_bp, a_bp, eeg, axis=1)
    eeg = sps.filtfilt(b_n, a_n, eeg, axis=1)
    return eeg.astype(np.float32)


def compute_band_powers(eeg: np.ndarray, fs: int = 256) -> np.ndarray:
    """Compute delta/theta/alpha/beta/gamma band powers per channel."""
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    powers = np.zeros((eeg.shape[0], len(bands)), dtype=np.float32)
    for i, ch in enumerate(eeg):
        freqs, psd = sps.welch(ch, fs=fs, nperseg=min(256, len(ch)))
        for j, (lo, hi) in enumerate(bands):
            powers[i, j] = psd[(freqs >= lo) & (freqs < hi)].sum()
    return powers


def _add_band_features(window: np.ndarray, fs: int) -> np.ndarray:
    """Append band power features to EEG window. Returns [19, T]."""
    powers = compute_band_powers(window, fs).mean(axis=0)  # [5]
    band_row = np.repeat(powers[:, np.newaxis], window.shape[1], axis=1)
    return np.vstack([window, band_row]).astype(np.float32)


# =============================================================================
# Dataset Loading
# =============================================================================

def load_preprocessed_dataset(
    root: str = "data/Preliminary_Preprocessed",
    overlap: float = DEFAULT_OVERLAP,
    target_sr: int = 256,
    max_trials_per_subject: int = 10,
):
    """
    Load EDF files and extract feature tensors.
    
    Returns:
        X: [N, 19, T] - 14 EEG channels + 5 band powers
        y: [N] class labels
        meta: list of (subject, word, trial_info)
        channels: channel names
        idx_to_label: class index -> word mapping
    """
    if mne is None:
        raise ImportError("Install MNE: pip install mne")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    # Load labels
    labels_path = root_path / "labels.json"
    labels = json.loads(labels_path.read_text())
    idx_to_label = {v: k for k, v in labels.items()}
    channels = _read_channels_file(root)

    trials_limit = max_trials_per_subject if max_trials_per_subject > 0 else None

    X, y, meta = [], [], []

    # Detect layout: subject/word/*.edf or word/*.edf
    subject_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    sample_word = next(iter(labels))
    by_subject = any((d / sample_word).is_dir() for d in subject_dirs)

    def process_edf(edf_path: Path, subject: str, class_idx: int, word: str):
        """Extract one full-trial sample from a single EDF file."""
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            trial_info = edf_path.stem.replace('_preprocessed', '')

            # Map channels
            raw_ch = [c.upper() for c in raw.ch_names]
            indices = [raw_ch.index(c.upper()) for c in channels]
            eeg = raw.get_data()[indices]

            # Downsample if needed
            orig_sr = int(raw.info['sfreq'])
            if orig_sr != target_sr:
                eeg = eeg[:, ::orig_sr // target_sr]

            # Use the full trial as one sample instead of slicing windows.
            # eeg = filter_eeg(eeg, target_sr)
            X.append(_add_band_features(eeg, target_sr))
            y.append(class_idx)
            meta.append((subject, word, trial_info))
        except Exception as e:
            print(f"Error processing {edf_path}: {e}")

    # Process files based on layout
    if by_subject:
        print("Loading subject-organized dataset...")
        for subj_dir in subject_dirs:
            for word, class_idx in labels.items():
                word_dir = subj_dir / word
                if not word_dir.exists():
                    continue
                edfs = sorted(word_dir.glob("*.edf"))
                if trials_limit:
                    edfs = edfs[:trials_limit]
                for edf in edfs:
                    process_edf(edf, subj_dir.name, class_idx, word)
    else:
        print("Loading word-organized dataset...")
        for word, class_idx in labels.items():
            word_dir = root_path / word
            if not word_dir.exists():
                continue
            edfs = sorted(word_dir.glob("*.edf"))
            if trials_limit:
                edfs = edfs[:trials_limit]
            for edf in edfs:
                process_edf(edf, "", class_idx, word)

    if not X:
        raise RuntimeError("No data loaded. Check EDF files.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {len(X)} samples, shape {X.shape}")
    
    return X, y, meta, channels, idx_to_label


# =============================================================================
# LOTO Splitting
# =============================================================================

def prepare_preprocessed_loto_pkl(
    root: str = "data/Preliminary_Preprocessed",
    seed: int = 1337,
    overlap: float = DEFAULT_OVERLAP,
    target_sr: int = 256,
    max_trials_per_subject: int = 10,
    train_trials_per_subject: int = 8,
    val_trials_per_subject: int = 1,
    test_trials_per_subject: int = 1,
    per_subject: bool = True,
):
    """
    Create train/val/test splits using Leave-One-Trial-Out.
    
    Args:
        per_subject: If True, creates separate splits for each subject.
    """
    root_path = Path(root)
    (root_path / "preprocessed_pkl").mkdir(parents=True, exist_ok=True)

    X, y, meta, channels, idx_to_label = load_preprocessed_dataset(
        root=root, overlap=overlap, target_sr=target_sr,
        max_trials_per_subject=max_trials_per_subject,
    )

    required = train_trials_per_subject + val_trials_per_subject + test_trials_per_subject

    # Build nested index: class -> subject -> trial -> [sample indices]
    trials_map: dict[int, dict[str, dict[str, list[int]]]] = {}
    for idx, (subj, _, trial) in enumerate(meta):
        cls = int(y[idx])
        trials_map.setdefault(cls, {}).setdefault(subj, {}).setdefault(trial, []).append(idx)

    subjects = sorted({m[0] for m in meta})
    print(f"\nFound {len(subjects)} subjects: {subjects}")

    # Verify sufficient trials
    for cls, subj_map in trials_map.items():
        for subj, trial_map in subj_map.items():
            if len(trial_map) < required:
                raise ValueError(f"Class {cls}, Subject {subj}: only {len(trial_map)} trials (need {required})")

    def save_split(name: str, indices: list[int], out_dir: Path):
        """Save a single split to pickle."""
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "X": X[indices], "y": y[indices],
            "meta": [meta[i] for i in indices],
            "channels": channels, "idx_to_label": idx_to_label,
            "sr": target_sr,
            "win_samples": int(X.shape[-1]) if len(indices) else None,
        }
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pkl.dump(data, f)

    def do_split(subj_filter: str = None) -> dict:
        """Perform LOTO split for one subject or all."""
        train_idx, val_idx, test_idx = [], [], []
        subjs = [subj_filter] if subj_filter else subjects

        for cls in sorted(trials_map.keys()):
            for subj in subjs:
                if subj not in trials_map[cls]:
                    continue
                trial_ids = list(trials_map[cls][subj].keys())

                # Deterministic shuffle
                rng = np.random.default_rng(seed + cls * 1000 + sum(ord(c) for c in subj))
                rng.shuffle(trial_ids)

                # Split trials
                train_t = trial_ids[:train_trials_per_subject]
                val_t = trial_ids[train_trials_per_subject:train_trials_per_subject + val_trials_per_subject]
                test_t = trial_ids[train_trials_per_subject + val_trials_per_subject:required]
                train_t.extend(trial_ids[required:])  # Extra trials go to training

                # Collect sample indices
                for t in train_t:
                    train_idx.extend(trials_map[cls][subj][t])
                for t in val_t:
                    val_idx.extend(trials_map[cls][subj][t])
                for t in test_t:
                    test_idx.extend(trials_map[cls][subj][t])

        # Verify no trial leakage
        train_trials = {meta[i][2] for i in train_idx}
        val_trials = {meta[i][2] for i in val_idx}
        test_trials = {meta[i][2] for i in test_idx}
        
        if train_trials & val_trials or train_trials & test_trials or val_trials & test_trials:
            raise RuntimeError("Trial leakage detected!")

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    stats = {}
    
    if per_subject:
        print("\n=== Creating per-subject splits ===")
        for subj in subjects:
            print(f"\nProcessing {subj}...")
            split = do_split(subj)
            out_dir = root_path / "preprocessed_pkl" / subj

            for name, indices in split.items():
                save_split(name, indices, out_dir)

            stats[subj] = {k: len(v) for k, v in split.items()}
            print(f"  {subj}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

        stats["subjects"] = subjects
        stats["per_subject"] = True
    else:
        print("\n=== Creating combined splits ===")
        split = do_split()
        out_dir = root_path / "preprocessed_pkl"

        for name, indices in split.items():
            save_split(name, indices, out_dir)

        stats = {k: len(v) for k, v in split.items()}
        stats["subjects"] = subjects
        stats["per_subject"] = False

    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess EEG data with LOTO splitting")
    parser.add_argument("--root", default="data/Preliminary_Preprocessed")
    parser.add_argument("--combined", action="store_true", help="Combine all subjects into one split")
    parser.add_argument("--max_trials", type=int, default=0,
                        help="Max trials per word per subject (0 = unlimited). Use 0 for datasets with 20 trials.")
    args = parser.parse_args()

    if os.path.exists(args.root):
        stats = prepare_preprocessed_loto_pkl(
            args.root,
            per_subject=not args.combined,
            max_trials_per_subject=args.max_trials,
        )
        print(f"\nStats: {stats}")
    else:
        print(f"Dataset not found: {args.root}")

    print("\nDone!")

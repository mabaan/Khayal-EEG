#!/usr/bin/env python3
"""Shared utilities for Stage 1 EEG classifiers.

This module centralizes dataset loading, trial grouping/splitting, augmentations,
basic metrics, and result serialization so multiple Stage 1 model variants can
reuse the same pipeline pieces.
"""

import json
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #


def _parse_trial_info(trial_info: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract session token (Cxx) and trial token (Txx) from metadata."""
    if not trial_info:
        return None, None
    parts = trial_info.split("_")
    session = next((p for p in parts if p.upper().startswith("C")), None)
    trial = next((p for p in parts if p.upper().startswith("T")), None)
    return session, trial


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pkl.load(f)


def load_full_dataset(root: str, subject: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, list, Dict[int, str]]:
    """Load and concatenate train/val/test splits into a full dataset."""
    root_path = Path(root)
    pkl_root = root_path / "preprocessed_pkl"
    if not pkl_root.exists():
        raise FileNotFoundError(f"Missing preprocessed_pkl at {pkl_root}")

    if subject:
        dirs = [pkl_root / subject]
    else:
        if (pkl_root / "train.pkl").exists():
            dirs = [pkl_root]
        else:
            dirs = [d for d in pkl_root.iterdir() if d.is_dir()]
            if not dirs:
                raise FileNotFoundError("No subject folders found in preprocessed_pkl")

    X_all, y_all, meta_all = [], [], []
    idx_to_label: Dict[int, str] = {}

    for d in dirs:
        for split in ("train", "val", "test"):
            pkl_path = d / f"{split}.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Missing split: {pkl_path}")
            data = _load_pkl(pkl_path)
            X_all.append(data["X"])
            y_all.append(data["y"])
            meta_all.extend(data.get("meta", []))
            if not idx_to_label and "idx_to_label" in data:
                idx_to_label = data["idx_to_label"]

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    y = np.concatenate(y_all, axis=0).astype(np.int64)

    if not idx_to_label:
        labels_path = root_path / "labels.json"
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            idx_to_label = {v: k for k, v in labels.items()}
        else:
            idx_to_label = {i: f"W{i+1}" for i in range(int(y.max()) + 1)}

    if len(meta_all) != len(X):
        raise ValueError("Meta length does not match X length. Rebuild PKLs with meta.")

    return X, y, meta_all, idx_to_label


def discover_subjects(root: str) -> List[str]:
    """Discover subject folders under preprocessed_pkl."""
    pkl_root = Path(root) / "preprocessed_pkl"
    if not pkl_root.exists():
        raise FileNotFoundError(f"Missing preprocessed_pkl at {pkl_root}")
    subjects = [d.name for d in pkl_root.iterdir() if d.is_dir()]
    return sorted(subjects)


# --------------------------------------------------------------------------- #
# Trial grouping and selection
# --------------------------------------------------------------------------- #


@dataclass
class Trial:
    key: Tuple[str, int, str, str]
    subject: str
    class_idx: int
    session: str
    trial_tag: str
    trial_num: Optional[int]
    indices: List[int]


def _parse_trial_num(trial_tag: Optional[str]) -> Optional[int]:
    if not trial_tag:
        return None
    digits = "".join([c for c in trial_tag if c.isdigit()])
    return int(digits) if digits.isdigit() else None


def build_trials(
    y: np.ndarray,
    meta: list,
) -> Tuple[List[Trial], Dict[Tuple[str, int], List[str]], Dict[Tuple[str, int], int]]:
    """Group sample indices by trial, return trials and session map per (subject, class)."""
    trial_map: Dict[Tuple[str, int, str, str], List[int]] = {}
    session_map: Dict[Tuple[str, int], set] = {}
    trial_count_map: Dict[Tuple[str, int], set] = {}

    for idx, (subject, _word, trial_info) in enumerate(meta):
        session, trial_tag = _parse_trial_info(trial_info)
        session = session or "UNKNOWN"
        trial_tag = trial_tag or f"T{idx}"
        class_idx = int(y[idx])

        key = (subject, class_idx, session, trial_tag)
        trial_map.setdefault(key, []).append(idx)
        session_map.setdefault((subject, class_idx), set()).add(session)
        trial_count_map.setdefault((subject, class_idx), set()).add(trial_tag)

    trials: List[Trial] = []
    for key, indices in trial_map.items():
        subject, class_idx, session, trial_tag = key
        trial_num = _parse_trial_num(trial_tag)
        trials.append(Trial(key, subject, class_idx, session, trial_tag, trial_num, indices))

    session_map_list = {k: sorted(list(v)) for k, v in session_map.items()}
    trial_counts = {k: len(v) for k, v in trial_count_map.items()}
    return trials, session_map_list, trial_counts


def _sort_sessions(sessions: List[str]) -> List[str]:
    def key_fn(s: str):
        num = "".join([c for c in s if c.isdigit()])
        return (s[:1], int(num) if num.isdigit() else 1_000_000, s)

    return sorted(sessions, key=key_fn)


def filter_trials_for_classifier(
    trials: List[Trial],
    session_map: Dict[Tuple[str, int], List[str]],
    trial_counts: Dict[Tuple[str, int], int],
    classifier: str,
) -> List[Trial]:
    """Select trials for classifier A (session-1) or B (session-2)."""
    selected: List[Trial] = []

    for t in trials:
        sessions = session_map.get((t.subject, t.class_idx), [t.session])
        total_trials = trial_counts.get((t.subject, t.class_idx), 0)

        # If no session tags, fall back to trial number split when 20 trials exist
        if len(sessions) <= 1:
            if total_trials > 10 and t.trial_num is not None:
                if classifier.upper() == "A" and t.trial_num <= 10:
                    selected.append(t)
                elif classifier.upper() != "A" and t.trial_num > 10:
                    selected.append(t)
            else:
                # Words with only 10 trials go to both classifiers
                selected.append(t)
            continue

        sess_sorted = _sort_sessions(sessions)
        if classifier.upper() == "A":
            chosen = sess_sorted[0]
        else:
            chosen = sess_sorted[1] if len(sess_sorted) > 1 else sess_sorted[0]

        if t.session == chosen:
            selected.append(t)

    return selected


# --------------------------------------------------------------------------- #
# Augmentations and preprocessing
# --------------------------------------------------------------------------- #


def time_shift(x: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    """Circular time shift by ±max_shift samples."""
    B, C, T = x.shape
    shift = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    idx = (torch.arange(T, device=x.device).unsqueeze(0) - shift.unsqueeze(1)) % T
    return x.gather(2, idx.unsqueeze(1).expand(B, C, T))


def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Zero out random channels."""
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask


def time_mask(x: torch.Tensor, max_len: int = 20, n_masks: int = 2) -> torch.Tensor:
    """SpecAugment-style time masking."""
    B, C, T = x.shape
    x = x.clone()
    for _ in range(n_masks):
        length = torch.randint(1, max_len + 1, (1,)).item()
        start = torch.randint(0, max(1, T - length), (B,), device=x.device)
        for b in range(B):
            x[b, :, start[b]:start[b] + length] = 0
    return x


def amplitude_scale(x: torch.Tensor, lo: float = 0.8, hi: float = 1.2) -> torch.Tensor:
    """Random amplitude scaling per sample."""
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(lo, hi)
    return x * scale


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, n_classes: int = 16):
    """MixUp augmentation returning blended inputs and soft targets."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, (lam, y, y[perm])


def _normalize_batch(xb: torch.Tensor) -> torch.Tensor:
    return (xb - xb.mean(dim=2, keepdim=True)) / (xb.std(dim=2, keepdim=True) + 1e-6)


def _sanitize_pooling(T: int, P1: int, P2: int) -> tuple[int, int]:
    """Ensure pooling sizes don't collapse temporal dimension."""
    t1 = max(T // P1, 1)
    if T < P1:
        P1 = max(1, T)
        t1 = max(T // P1, 1)
    if P2 > t1:
        P2 = max(1, t1)
    while t1 // P2 == 0 and P2 > 1:
        P2 //= 2
    return P1, max(P2, 1)


# --------------------------------------------------------------------------- #
# Metrics and reporting
# --------------------------------------------------------------------------- #


def majority_vote(labels: List[int]) -> int:
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def confusion_from_trials(
    trials: List[Trial],
    test_indices: np.ndarray,
    sample_logits: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Build confusion matrix using average logits per trial (not majority vote)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    logit_map = {int(idx): logit for idx, logit in zip(test_indices, sample_logits)}

    for t in trials:
        logits = np.stack([logit_map[i] for i in t.indices], axis=0)
        avg_logit = logits.mean(axis=0)
        pred_label = int(np.argmax(avg_logit))
        cm[t.class_idx, pred_label] += 1

    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm.astype(np.float32) / row_sums


def compute_metrics(cm: np.ndarray) -> Dict[str, float]:
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(float(f1))

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def save_loss_plot(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    out_path: Path,
    title: str = "Training Loss",
) -> None:
    """Save train/val loss curves as a PNG to *out_path*."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train")
    if val_losses:
        clean = [v for v in val_losses if v == v]  # drop NaN
        if clean:
            ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_results(
    out_dir: Path,
    prefix: str,
    cm: np.ndarray,
    fold_accs: Optional[List[float]] = None,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_norm = row_normalize(cm)
    metrics = compute_metrics(cm)

    payload = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": cm_norm.tolist(),
        "metrics": metrics,
    }
    if fold_accs is not None:
        payload["fold_accuracy"] = [float(v) for v in fold_accs]

    out_path = out_dir / f"{prefix}_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return metrics


__all__ = [
    # dataset
    "load_full_dataset",
    "discover_subjects",
    # trial utilities
    "Trial",
    "build_trials",
    "filter_trials_for_classifier",
    # augmentations / preprocessing
    "time_shift",
    "channel_dropout",
    "time_mask",
    "amplitude_scale",
    "mixup",
    "_normalize_batch",
    "_sanitize_pooling",
    # metrics / reporting
    "confusion_from_trials",
    "row_normalize",
    "compute_metrics",
    "save_results",
    "save_loss_plot",
    "majority_vote",
]

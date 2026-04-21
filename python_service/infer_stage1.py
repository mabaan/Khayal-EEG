from __future__ import annotations

from typing import Dict, List

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def infer_stage1_posteriors(slot_tensors: List[np.ndarray], label_ids: List[int]) -> List[Dict[str, object]]:
    posteriors: List[Dict[str, object]] = []

    for index, tensor in enumerate(slot_tensors):
        slot = index + 1
        energy = float(np.mean(np.abs(tensor[:14, :])))
        seed = int(abs(energy) * 1000) + slot * 97

        scores = np.array(
            [
                np.sin((seed + label_id * 13) * 0.017) + np.cos((seed + label_id * 29) * 0.011)
                for label_id in label_ids
            ]
        )
        probs = _softmax(scores)

        posteriors.append(
            {
                "slot": slot,
                "probabilities": {str(label_id): float(prob) for label_id, prob in zip(label_ids, probs)},
            }
        )

    return posteriors

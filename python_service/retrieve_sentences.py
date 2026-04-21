from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from .config import SENTENCE_CATALOG_PATH, STAGE2_TOP_K


def load_sentence_catalog() -> List[Dict[str, object]]:
    payload = json.loads(Path(SENTENCE_CATALOG_PATH).read_text(encoding="utf-8"))
    return payload["sentence_catalog"]


def retrieve_candidates(
    stage1_posteriors: List[Dict[str, object]],
    top_k: int = STAGE2_TOP_K,
) -> List[Dict[str, object]]:
    catalog = load_sentence_catalog()
    epsilon = 1e-9

    scored: List[Dict[str, object]] = []
    for sentence in catalog:
        word_ids = sentence["word_ids"]
        score = 0.0
        for idx, word_id in enumerate(word_ids):
            probs: Dict[str, float] = stage1_posteriors[idx]["probabilities"]
            score += math.log(max(probs.get(str(word_id), epsilon), epsilon))

        scored.append(
            {
                "sentence_id": int(sentence["sentence_id"]),
                "arabic": sentence["arabic"],
                "score": float(score),
                "word_ids": word_ids,
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["sentence_id"]))
    return scored[:top_k]

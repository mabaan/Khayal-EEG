from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_GENERATE_ENDPOINT, OLLAMA_MODEL, RERANK_TEMPERATURE


PROMPT_TEMPLATE = """
You are a deterministic reranker for imagined speech decoding.
Pick one candidate index based on the 3-slot posterior evidence.
Return only JSON: {"selected_index": <1-based-index>}.

Candidates:
{candidates}

Stage1 posteriors:
{posteriors}
""".strip()


def _parse_index(text: str, candidate_count: int) -> int | None:
    try:
        parsed = json.loads(text)
        idx = int(parsed.get("selected_index"))
        if 1 <= idx <= candidate_count:
            return idx
    except Exception:
        pass

    match = re.search(r"(\d+)", text)
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= candidate_count:
            return idx

    return None


def rerank_with_qwen(
    candidates: List[Dict[str, object]],
    stage1_posteriors: List[Dict[str, object]],
) -> Tuple[Dict[str, object], bool, str]:
    if not candidates:
        raise ValueError("No retrieval candidates available.")

    prompt = PROMPT_TEMPLATE.format(
        candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        posteriors=json.dumps(stage1_posteriors, ensure_ascii=False, indent=2),
    )

    request_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": RERANK_TEMPERATURE},
    }

    raw_text = ""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}{OLLAMA_GENERATE_ENDPOINT}",
            json=request_payload,
            timeout=25,
        )
        if response.ok:
            data = response.json()
            raw_text = str(data.get("response", ""))
            parsed_idx = _parse_index(raw_text, len(candidates))
            if parsed_idx is not None:
                return candidates[parsed_idx - 1], False, raw_text
    except Exception:
        raw_text = ""

    return candidates[0], True, raw_text

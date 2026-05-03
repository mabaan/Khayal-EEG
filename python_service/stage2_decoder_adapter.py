from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
import torch

from .config import (
    LABELS_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_HEALTH_ENDPOINT,
    OLLAMA_MODEL,
    RERANK_TEMPERATURE,
    SENTENCE_CATALOG_PATH,
    STAGE2_CACHE_DIR,
    STAGE2_EMBEDDING_MODEL,
    STAGE2_HYBRID_ALPHA,
    STAGE2_HYBRID_NORMALIZATION,
    STAGE2_RETRIEVAL_MODE,
)
from .research_adapters.stage2_llm_rag import ProbabilitySentenceRetriever


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None


def cuda_available() -> bool:
    return torch.cuda.is_available()


def _has_local_hf_assets(model_name: str) -> bool:
    candidate = Path(model_name)
    if candidate.exists():
        return True

    repo_dir_name = f"models--{model_name.replace('/', '--')}"
    cache_roots = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    cache_roots.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".huggingface" / "hub",
        ]
    )
    return any((root / repo_dir_name).exists() for root in cache_roots)


def ollama_available(timeout_seconds: float = 1.5) -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}{OLLAMA_HEALTH_ENDPOINT}", timeout=timeout_seconds)
        return response.ok
    except Exception:
        return False


def _normalize_probs(values: Sequence[float]) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        return np.full((len(probs),), 1.0 / max(1, len(probs)), dtype=np.float32)
    return (probs / total).astype(np.float32)


def _parse_choice(text: str, candidate_count: int) -> int | None:
    try:
        parsed = json.loads(text)
        selected_index = int(parsed.get("selected_index"))
        if 1 <= selected_index <= candidate_count:
            return selected_index
    except Exception:
        pass

    match = re.search(r"(\d+)", text)
    if not match:
        return None
    selected_index = int(match.group(1))
    if 1 <= selected_index <= candidate_count:
        return selected_index
    return None


class Stage2DecoderAdapter:
    def __init__(self, labels_path: str | Path = LABELS_PATH, sentence_catalog_path: str | Path = SENTENCE_CATALOG_PATH):
        self.labels = sorted(_load_json(Path(labels_path)).get("labels", []), key=lambda item: int(item["id"]))
        if len(self.labels) != 25:
            raise ValueError("labels.json must contain 25 entries.")
        self.sentence_catalog = _load_json(Path(sentence_catalog_path)).get("sentence_catalog", [])
        if len(self.sentence_catalog) != 12:
            raise ValueError("sentence_catalog.json must contain 12 sentences.")
        self.label_order = [str(item["word"]) for item in self.labels]
        self.device = "cuda" if cuda_available() else "cpu"

    def _posterior_only_candidates(self, position_prob_vectors: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        epsilon = 1e-12
        candidates: List[Dict[str, Any]] = []
        for sentence in self.sentence_catalog:
            word_ids = [int(word_id) for word_id in sentence["word_ids"]]
            slot_probs = [float(position_prob_vectors[position][word_id]) for position, word_id in enumerate(word_ids)]
            posterior_score = sum(math.log(max(prob, epsilon)) for prob in slot_probs)
            candidates.append(
                {
                    "sentence_id": str(sentence["sentence_id"]),
                    "arabic": str(sentence["arabic"]),
                    "romanized": str(sentence["romanized"]),
                    "english": str(sentence.get("english", "")),
                    "word_ids": word_ids,
                    "word_probabilities": slot_probs,
                    "posterior_score": float(posterior_score),
                    "transformer_score": None,
                    "retrieval_score": float(posterior_score),
                }
            )
        candidates.sort(key=lambda item: (-float(item["retrieval_score"]), item["sentence_id"]))
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = int(rank)
        return candidates

    def _build_slot_topk_prompt_rows(self, stage1_slots: Sequence[Dict[str, Any]]) -> List[str]:
        rows = []
        for slot in stage1_slots:
            entries = []
            for item in slot.get("top_k", []):
                entries.append(f"{item['word']} / {item['arabic']} = {float(item['probability']):.4f}")
            rows.append(f"Slot {slot['slot']}: " + ", ".join(entries))
        return rows

    def _retrieve_candidates(
        self,
        position_prob_vectors: Sequence[np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], List[str], bool]:
        base_candidates = self._posterior_only_candidates(position_prob_vectors)
        warnings: List[str] = []

        if self.device != "cuda":
            warnings.append("CUDA is unavailable; transformer retrieval is running on CPU.")

        if not transformers_available():
            warnings.append("transformers is not installed; using posterior-only shortlist.")
            return base_candidates, warnings, False

        if not _has_local_hf_assets(STAGE2_EMBEDDING_MODEL):
            warnings.append("Local transformer retrieval assets were not found; using posterior-only shortlist.")
            return base_candidates, warnings, False

        sentence_catalog = [
            {
                "sentence_id": str(item["sentence_id"]),
                "words": list(item["word_tokens"]),
                "class_indices": [int(word_id) for word_id in item["word_ids"]],
            }
            for item in self.sentence_catalog
        ]

        try:
            STAGE2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = STAGE2_CACHE_DIR / "retriever_cache.npz"
            retriever = ProbabilitySentenceRetriever(
                embedding_model_name=STAGE2_EMBEDDING_MODEL,
                label_order=self.label_order,
                sentence_catalog=sentence_catalog,
                device=self.device,
                hf_timeout=10,
                hf_retries=1,
                local_only=True,
                cache_path=str(cache_path),
            )
            retrieval = retriever.retrieve(
                position_prob_vectors=position_prob_vectors,
                top_k=len(sentence_catalog),
                retrieval_mode=STAGE2_RETRIEVAL_MODE,
                hybrid_alpha=STAGE2_HYBRID_ALPHA,
                hybrid_normalization=STAGE2_HYBRID_NORMALIZATION,
            )
            ranked_candidates = {
                str(item["sentence_id"]): item
                for item in retrieval["ranked_candidates"]
            }
            combined: List[Dict[str, Any]] = []
            for candidate in base_candidates:
                retrieved = ranked_candidates.get(candidate["sentence_id"])
                updated = dict(candidate)
                if retrieved is not None:
                    updated["transformer_score"] = float(retrieved["normalized_cosine_score"])
                    updated["retrieval_score"] = float(retrieved["retrieval_score"])
                combined.append(updated)
            combined.sort(key=lambda item: (-float(item["retrieval_score"]), item["sentence_id"]))
            for rank, candidate in enumerate(combined, start=1):
                candidate["rank"] = int(rank)
            return combined, warnings, True
        except Exception as exc:
            warnings.append(f"Transformer retrieval failed: {exc}")
            return base_candidates, warnings, False

    def _select_with_qwen(
        self,
        shortlist: List[Dict[str, Any]],
        stage1_slots: Sequence[Dict[str, Any]],
    ) -> Tuple[int, bool, str | None, List[str]]:
        warnings: List[str] = []
        if not ollama_available():
            warnings.append("Qwen/Ollama is unavailable; using retrieval rank-1 fallback.")
            return 1, True, None, warnings

        slot_rows = "\n".join(self._build_slot_topk_prompt_rows(stage1_slots))
        candidate_rows = "\n".join(
            [
                (
                    f"{candidate['rank']}. {candidate['sentence_id']} | {candidate['arabic']} | "
                    f"{candidate['romanized']} | hybrid_score={float(candidate['retrieval_score']):.4f}"
                )
                for candidate in shortlist
            ]
        )
        prompt = (
            "Select the single best sentence candidate from imagined-speech EEG evidence.\n"
            "The shortlist is already ranked by transformer retrieval. Use the slot posteriors as the deciding evidence.\n"
            "Return only JSON in the form {\"selected_index\": <n>}.\n\n"
            f"Stage 1 slot evidence:\n{slot_rows}\n\n"
            f"Candidate shortlist:\n{candidate_rows}\n"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": RERANK_TEMPERATURE},
        }
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}{OLLAMA_GENERATE_ENDPOINT}",
                json=payload,
                timeout=40,
            )
            response.raise_for_status()
            raw_output = str(response.json().get("response", "")).strip()
            selected_index = _parse_choice(raw_output, len(shortlist))
            if selected_index is None:
                warnings.append("Qwen/Ollama returned an invalid candidate index; using retrieval rank-1 fallback.")
                return 1, True, raw_output, warnings
            return selected_index, False, raw_output, warnings
        except Exception as exc:
            warnings.append(f"Qwen/Ollama selection failed: {exc}")
            return 1, True, None, warnings

    def decode(
        self,
        stage1_slots: Sequence[Dict[str, Any]],
        retrieval_topk: int = 5,
        stage2_mode: str = "qwen",
    ) -> Dict[str, Any]:
        mode = str(stage2_mode).strip().lower()
        if mode != "qwen":
            raise ValueError(f"Unsupported Stage 2 mode: {stage2_mode!r}")

        position_prob_vectors = [
            _normalize_probs([float(slot["probabilities"][str(index)]) for index in range(25)])
            for slot in stage1_slots
        ]

        candidates, retrieval_warnings, transformer_retrieval_used = self._retrieve_candidates(position_prob_vectors)
        shortlist = [dict(candidate) for candidate in candidates[: max(1, int(retrieval_topk))]]
        selected_index, used_fallback, raw_llm_output, qwen_warnings = self._select_with_qwen(shortlist, stage1_slots)
        warnings = retrieval_warnings + qwen_warnings

        selected_index = max(1, min(int(selected_index), len(shortlist)))
        selected_candidate = dict(shortlist[selected_index - 1])
        selected_candidate["rerank_selected"] = True

        ui_candidates = []
        for candidate in shortlist:
            ui_candidate = dict(candidate)
            ui_candidate["rerank_selected"] = candidate["sentence_id"] == selected_candidate["sentence_id"]
            ui_candidates.append(ui_candidate)

        return {
            "mode": mode,
            "retrieval_topk": int(retrieval_topk),
            "used_fallback": bool(used_fallback),
            "candidate_sentences": ui_candidates,
            "raw_llm_output": raw_llm_output,
            "warnings": warnings,
            "reranker_model": OLLAMA_MODEL,
            "transformer_model": STAGE2_EMBEDDING_MODEL,
            "device": self.device,
            "transformer_retrieval_used": transformer_retrieval_used,
            "prediction": {
                "sentence_id": str(selected_candidate["sentence_id"]),
                "arabic": str(selected_candidate["arabic"]),
                "romanized": str(selected_candidate["romanized"]),
                "english": str(selected_candidate.get("english", "")),
                "score": float(selected_candidate["retrieval_score"]),
            },
        }

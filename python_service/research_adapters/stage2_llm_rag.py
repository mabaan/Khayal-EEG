#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 method 4: retrieval-augmented LLM reranking over sentence-trial EEG evidence.
"""

import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        if probs.size == 0:
            return probs
        return np.full((probs.size,), 1.0 / float(probs.size), dtype=np.float32)
    return (probs / total).astype(np.float32)


def l2_normalize(vec: np.ndarray, axis: int = -1) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    denom = np.linalg.norm(vec, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (vec / denom).astype(np.float32)


def safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def safe_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_val = float(values.min())
    max_val = float(values.max())
    if (max_val - min_val) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_val) / (max_val - min_val)).astype(np.float32)


def normalize_score_vector(values: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "none":
        return np.asarray(values, dtype=np.float32)
    if mode == "zscore":
        return safe_zscore(values)
    if mode == "minmax":
        return safe_minmax(values)
    raise ValueError(
        f"Unsupported hybrid normalization '{mode}'. Use 'none', 'zscore', or 'minmax'."
    )


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_sentence_catalog(corpus, label_to_idx: Dict[str, int]) -> List[Dict[str, object]]:
    catalog: List[Dict[str, object]] = []
    for sentence_id in corpus.sentence_ids:
        words = list(corpus.get_sentence(sentence_id))
        catalog.append(
            {
                "sentence_id": str(sentence_id),
                "words": words,
                "class_indices": [int(label_to_idx[word]) for word in words],
            }
        )
    return catalog


def extract_topk_predictions(
    probs: np.ndarray,
    idx_to_label: Dict[int, str],
    topk: int,
) -> List[Dict[str, object]]:
    probs = normalize_probs(probs)
    k_eff = min(int(topk), int(probs.shape[0]))
    top_idx = np.argsort(-probs)[:k_eff]
    entries: List[Dict[str, object]] = []
    for idx in top_idx:
        prob = float(probs[int(idx)])
        entries.append(
            {
                "index": int(idx),
                "word": str(idx_to_label[int(idx)]),
                "probability": prob,
                "log_probability": float(math.log(max(prob, 1e-12))),
            }
        )
    return entries


def _compute_metrics(
    predictions: List[Dict[str, object]],
    retrieval_topk: Optional[int] = None,
) -> Dict[str, float]:
    total_trials = len(predictions)
    sentence_correct = sum(1 for item in predictions if bool(item.get("correct")))

    word_errors = 0
    total_words = 0
    position_correct = [0, 0, 0]
    retrieval_ranks: List[float] = []
    retrieval_hits_at_1 = 0
    retrieval_hits_at_3 = 0
    retrieval_hits_at_5 = 0
    retrieval_hits_at_k = 0
    retrieval_only_correct = 0
    parse_success = 0
    fallback_count = 0
    selected_ranks: List[float] = []

    for item in predictions:
        pred_words = list(item.get("predicted", []))
        true_words = list(item.get("true", []))
        for pos, (pred_word, true_word) in enumerate(zip(pred_words, true_words)):
            total_words += 1
            if pred_word == true_word:
                position_correct[pos] += 1
            else:
                word_errors += 1

        retrieval = item.get("retrieval", {})
        true_rank = retrieval.get("true_sentence_rank")
        if true_rank is not None:
            true_rank = int(true_rank)
            retrieval_ranks.append(float(true_rank))
            retrieval_hits_at_1 += int(true_rank <= 1)
            retrieval_hits_at_3 += int(true_rank <= 3)
            retrieval_hits_at_5 += int(true_rank <= 5)
            if retrieval_topk is not None:
                retrieval_hits_at_k += int(true_rank <= int(retrieval_topk))

        retrieval_only_correct += int(bool(retrieval.get("top1_correct")))

        reranker = item.get("reranker", {})
        used_fallback = bool(reranker.get("used_fallback"))
        parse_success += int(not used_fallback)
        fallback_count += int(used_fallback)

        selected_candidate = item.get("selected_candidate", {})
        selected_rank = selected_candidate.get("retrieval_rank")
        if selected_rank is not None:
            selected_ranks.append(float(selected_rank))

    accuracy = (sentence_correct / float(total_trials)) if total_trials else 0.0
    wer = (word_errors / float(total_words)) if total_words else 0.0
    mrr = (
        sum(1.0 / rank for rank in retrieval_ranks) / float(len(retrieval_ranks))
        if retrieval_ranks else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "sentence_accuracy": accuracy,
        "wer": wer,
        "word_error_rate": wer,
        "retrieval_recall_at_1": (
            retrieval_hits_at_1 / float(total_trials) if total_trials else 0.0
        ),
        "retrieval_recall_at_3": (
            retrieval_hits_at_3 / float(total_trials) if total_trials else 0.0
        ),
        "retrieval_recall_at_5": (
            retrieval_hits_at_5 / float(total_trials) if total_trials else 0.0
        ),
        "retrieval_mrr": mrr,
        "retrieval_only_accuracy": (
            retrieval_only_correct / float(total_trials) if total_trials else 0.0
        ),
        "reranker_parse_success_rate": (
            parse_success / float(total_trials) if total_trials else 0.0
        ),
        "reranker_fallback_rate": (
            fallback_count / float(total_trials) if total_trials else 0.0
        ),
        "mean_selected_retrieval_rank": (
            sum(selected_ranks) / float(len(selected_ranks)) if selected_ranks else 0.0
        ),
        "position_1_error_rate": (
            1.0 - (position_correct[0] / float(total_trials)) if total_trials else 0.0
        ),
        "position_2_error_rate": (
            1.0 - (position_correct[1] / float(total_trials)) if total_trials else 0.0
        ),
        "position_3_error_rate": (
            1.0 - (position_correct[2] / float(total_trials)) if total_trials else 0.0
        ),
    }
    if retrieval_topk is not None:
        metrics[f"retrieval_recall_at_{int(retrieval_topk)}"] = (
            retrieval_hits_at_k / float(total_trials) if total_trials else 0.0
        )
    return metrics


class TransformerTextEmbedder:
    """Mean-pooling transformer embedder for vocabulary and sentence retrieval."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        hf_timeout: int = 120,
        hf_retries: int = 3,
        local_only: bool = False,
        max_length: int = 128,
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)
        self.hf_timeout = int(hf_timeout)
        self.hf_retries = int(hf_retries)
        self.local_only = bool(local_only)
        self.max_length = int(max_length)

        self.tokenizer = None
        self.model = None

    def _load_with_retry(self, fn, *args, **kwargs):
        last_err = None
        for attempt in range(1, self.hf_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover
                last_err = exc
                wait_seconds = min(30, 5 * attempt)
                print(
                    f"[RAG embedder] load failed (attempt {attempt}/{self.hf_retries}): {exc}; "
                    f"retrying in {wait_seconds}s...",
                    flush=True,
                )
                import time

                time.sleep(wait_seconds)
        raise last_err

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install transformers: pip install transformers") from exc

        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(self.hf_timeout))
        os.environ.setdefault("HF_HUB_OFFLINE", "1" if self.local_only else "0")

        print(f"Loading retrieval embedder: {self.model_name}")
        self.tokenizer = self._load_with_retry(
            AutoTokenizer.from_pretrained,
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_only,
        )
        self.model = self._load_with_retry(
            AutoModel.from_pretrained,
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_only,
            torch_dtype=torch.float32,
        )

        target_device = torch.device(
            self.device if torch.cuda.is_available() and str(self.device).lower().startswith("cuda")
            else "cpu"
        )
        self.model = self.model.to(target_device)
        self.model.eval()
        print("Retrieval embedder loaded successfully.")

    def _input_device(self) -> torch.device:
        if self.model is None:
            return torch.device(
                self.device if torch.cuda.is_available() and str(self.device).lower().startswith("cuda")
                else "cpu"
            )
        for param in self.model.parameters():
            return param.device
        return torch.device(
            self.device if torch.cuda.is_available() and str(self.device).lower().startswith("cuda")
            else "cpu"
        )

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._load_model()
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        device = self._input_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype(np.float32)


class ProbabilitySentenceRetriever:
    """Build soft EEG queries and retrieve sentence candidates from the fixed corpus."""

    def __init__(
        self,
        embedding_model_name: str,
        label_order: Sequence[str],
        sentence_catalog: List[Dict[str, object]],
        device: str = "cuda",
        hf_timeout: int = 120,
        hf_retries: int = 3,
        local_only: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
        self.embedding_model_name = str(embedding_model_name)
        self.label_order = [str(item) for item in label_order]
        self.sentence_catalog = [
            {
                "sentence_id": str(item["sentence_id"]),
                "words": [str(word) for word in item["words"]],
                "class_indices": [int(idx) for idx in item["class_indices"]],
            }
            for item in sentence_catalog
        ]
        self.cache_path = Path(cache_path) if cache_path else None
        self.embedder = TransformerTextEmbedder(
            model_name=self.embedding_model_name,
            device=device,
            hf_timeout=hf_timeout,
            hf_retries=hf_retries,
            local_only=local_only,
        )

        self.word_embeddings: Optional[np.ndarray] = None
        self.sentence_vectors: Optional[np.ndarray] = None

    def _metadata_payload(self) -> Dict[str, object]:
        return {
            "embedding_model_name": self.embedding_model_name,
            "label_order": list(self.label_order),
            "sentence_catalog": [
                {
                    "sentence_id": item["sentence_id"],
                    "words": list(item["words"]),
                    "class_indices": list(item["class_indices"]),
                }
                for item in self.sentence_catalog
            ],
        }

    def _load_cache(self) -> bool:
        if self.cache_path is None or not self.cache_path.exists():
            return False

        try:
            payload = np.load(self.cache_path, allow_pickle=False)
            metadata_json = str(payload["metadata_json"].item())
            metadata = json.loads(metadata_json)
            if metadata != self._metadata_payload():
                return False
            self.word_embeddings = payload["word_embeddings"].astype(np.float32)
            self.sentence_vectors = payload["sentence_vectors"].astype(np.float32)
            print(f"Loaded RAG embedding cache from {self.cache_path}")
            return True
        except Exception as exc:
            print(f"[WARN] Could not load embedding cache {self.cache_path}: {exc}")
            return False

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        if self.word_embeddings is None or self.sentence_vectors is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            metadata_json=np.array(json.dumps(self._metadata_payload(), ensure_ascii=False)),
            word_embeddings=self.word_embeddings.astype(np.float32),
            sentence_vectors=self.sentence_vectors.astype(np.float32),
        )
        print(f"Saved RAG embedding cache to {self.cache_path}")

    def _ensure_ready(self) -> None:
        if self.word_embeddings is not None and self.sentence_vectors is not None:
            return
        if self._load_cache():
            return

        self.word_embeddings = self.embedder.encode(self.label_order).astype(np.float32)
        if self.word_embeddings.shape[0] != len(self.label_order):
            raise ValueError(
                f"Embedding model returned {self.word_embeddings.shape[0]} word vectors "
                f"for {len(self.label_order)} labels."
            )
        self.word_embeddings = l2_normalize(self.word_embeddings, axis=1)

        sentence_vectors: List[np.ndarray] = []
        for item in self.sentence_catalog:
            class_indices = list(item["class_indices"])
            slot_vectors = [self.word_embeddings[int(class_idx)] for class_idx in class_indices]
            sentence_vec = np.concatenate(slot_vectors, axis=0).astype(np.float32)
            sentence_vectors.append(sentence_vec)

        self.sentence_vectors = l2_normalize(np.stack(sentence_vectors, axis=0), axis=1)
        self._save_cache()

    def build_query_vector(self, position_prob_vectors: Sequence[np.ndarray]) -> np.ndarray:
        self._ensure_ready()
        if len(position_prob_vectors) != 3:
            raise ValueError("Expected exactly 3 position probability vectors.")

        assert self.word_embeddings is not None
        slot_vectors: List[np.ndarray] = []
        for probs in position_prob_vectors:
            probs = normalize_probs(np.asarray(probs, dtype=np.float32))
            if probs.shape[0] != self.word_embeddings.shape[0]:
                raise ValueError(
                    f"Probability vector has {probs.shape[0]} classes but the retriever expects "
                    f"{self.word_embeddings.shape[0]} labels."
                )
            slot_vectors.append(np.matmul(probs, self.word_embeddings).astype(np.float32))

        query = np.concatenate(slot_vectors, axis=0).astype(np.float32)
        return l2_normalize(query)

    def retrieve(
        self,
        position_prob_vectors: Sequence[np.ndarray],
        top_k: int,
        retrieval_mode: str = "hybrid",
        hybrid_alpha: float = 0.5,
        hybrid_normalization: str = "zscore",
    ) -> Dict[str, object]:
        self._ensure_ready()
        retrieval_mode = str(retrieval_mode).lower()
        top_k = max(1, int(top_k))
        hybrid_alpha = float(hybrid_alpha)

        if retrieval_mode not in {"cosine", "eeg", "hybrid"}:
            raise ValueError(
                f"Unsupported retrieval mode '{retrieval_mode}'. "
                "Use 'cosine', 'eeg', or 'hybrid'."
            )

        query_vector = self.build_query_vector(position_prob_vectors)
        assert self.sentence_vectors is not None
        cosine_scores = np.matmul(self.sentence_vectors, query_vector).astype(np.float32)

        eeg_logprobs: List[float] = []
        normalized_prob_vectors = [
            normalize_probs(np.asarray(probs, dtype=np.float32)) for probs in position_prob_vectors
        ]
        for item in self.sentence_catalog:
            class_indices = list(item["class_indices"])
            eeg_score = 0.0
            for pos_idx, class_idx in enumerate(class_indices):
                prob = float(normalized_prob_vectors[pos_idx][int(class_idx)])
                eeg_score += float(math.log(max(prob, 1e-12)))
            eeg_logprobs.append(float(eeg_score))

        eeg_scores = np.asarray(eeg_logprobs, dtype=np.float32)
        norm_cosine = normalize_score_vector(cosine_scores, hybrid_normalization)
        norm_eeg = normalize_score_vector(eeg_scores, hybrid_normalization)

        if retrieval_mode == "cosine":
            retrieval_scores = norm_cosine
        elif retrieval_mode == "eeg":
            retrieval_scores = norm_eeg
        else:
            retrieval_scores = (
                float(hybrid_alpha) * norm_cosine
                + (1.0 - float(hybrid_alpha)) * norm_eeg
            ).astype(np.float32)

        ranking = np.argsort(-retrieval_scores)
        top_k_eff = min(top_k, int(len(ranking)))

        ranked_candidates: List[Dict[str, object]] = []
        for rank, idx in enumerate(ranking, start=1):
            item = self.sentence_catalog[int(idx)]
            class_indices = list(item["class_indices"])
            candidate_slot_probs = [
                float(normalized_prob_vectors[pos_idx][int(class_idx)])
                for pos_idx, class_idx in enumerate(class_indices)
            ]
            ranked_candidates.append(
                {
                    "sentence_id": str(item["sentence_id"]),
                    "words": list(item["words"]),
                    "class_indices": class_indices,
                    "retrieval_rank": int(rank),
                    "cosine_score": float(cosine_scores[int(idx)]),
                    "eeg_logprob": float(eeg_scores[int(idx)]),
                    "normalized_cosine_score": float(norm_cosine[int(idx)]),
                    "normalized_eeg_logprob": float(norm_eeg[int(idx)]),
                    "retrieval_score": float(retrieval_scores[int(idx)]),
                    "candidate_word_probabilities": candidate_slot_probs,
                }
            )

        return {
            "query_vector_norm": float(np.linalg.norm(query_vector)),
            "retrieval_mode": retrieval_mode,
            "hybrid_alpha": float(hybrid_alpha),
            "hybrid_normalization": str(hybrid_normalization),
            "candidate_count": int(len(ranked_candidates)),
            "ranked_candidates": ranked_candidates,
            "retrieved_candidates": ranked_candidates[:top_k_eff],
        }


class JsonPromptCache:
    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path) if path else None
        self.payload: Dict[str, object] = {"responses": {}}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                self.payload = json.load(handle)
            if not isinstance(self.payload, dict) or "responses" not in self.payload:
                self.payload = {"responses": {}}
        except Exception as exc:
            print(f"[WARN] Could not read LLM prompt cache {self.path}: {exc}")
            self.payload = {"responses": {}}

    def get(self, key: str) -> Optional[Dict[str, object]]:
        responses = self.payload.get("responses", {})
        if not isinstance(responses, dict):
            return None
        value = responses.get(str(key))
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: Dict[str, object]) -> None:
        responses = self.payload.setdefault("responses", {})
        if not isinstance(responses, dict):
            self.payload["responses"] = {}
            responses = self.payload["responses"]
        responses[str(key)] = value
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, ensure_ascii=False)


class LLMCandidateReranker:
    """Constrained reranker that selects one sentence index from a retrieved shortlist."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        hf_timeout: int = 120,
        hf_retries: int = 3,
        local_only: bool = False,
        cache_path: Optional[str] = None,
        max_new_tokens: int = 8,
        arabic_word_map: Optional[Dict[str, str]] = None,
        prompt_language: str = "arabic",
        surface_form: str = "arabic",
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)
        self.hf_timeout = int(hf_timeout)
        self.hf_retries = int(hf_retries)
        self.local_only = bool(local_only)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_language = str(prompt_language).strip().lower()
        if self.prompt_language not in {"english", "arabic"}:
            raise ValueError(
                f"Unsupported prompt_language={prompt_language!r}; expected 'english' or 'arabic'."
            )
        self.surface_form = str(surface_form).strip().lower()
        if self.surface_form not in {"romanized", "arabic"}:
            raise ValueError(
                f"Unsupported surface_form={surface_form!r}; expected 'romanized' or 'arabic'."
            )
        self.arabic_word_map = {
            str(word): str(surface)
            for word, surface in (arabic_word_map or {}).items()
        }

        self.cache = JsonPromptCache(cache_path)
        self.tokenizer = None
        self.model = None

    def _load_with_retry(self, fn, *args, **kwargs):
        last_err = None
        for attempt in range(1, self.hf_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover
                last_err = exc
                wait_seconds = min(30, 5 * attempt)
                print(
                    f"[LLM reranker] load failed (attempt {attempt}/{self.hf_retries}): {exc}; "
                    f"retrying in {wait_seconds}s...",
                    flush=True,
                )
                import time

                time.sleep(wait_seconds)
        raise last_err

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install transformers: pip install transformers") from exc

        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(self.hf_timeout))
        os.environ.setdefault("HF_HUB_OFFLINE", "1" if self.local_only else "0")

        print(f"Loading LLM reranker: {self.model_name}")
        self.tokenizer = self._load_with_retry(
            AutoTokenizer.from_pretrained,
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_only,
        )
        self.model = self._load_with_retry(
            AutoModelForCausalLM.from_pretrained,
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_only,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() and str(self.device).lower().startswith("cuda") else None,
        )

        if not torch.cuda.is_available() or not str(self.device).lower().startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("LLM reranker loaded successfully.")

    def _input_device(self) -> torch.device:
        if self.model is None:
            return torch.device(
                self.device if torch.cuda.is_available() and str(self.device).lower().startswith("cuda")
                else "cpu"
            )
        for param in self.model.parameters():
            return param.device
        return torch.device(
            self.device if torch.cuda.is_available() and str(self.device).lower().startswith("cuda")
            else "cpu"
        )

    def _word_surface(self, word: str) -> str:
        word = str(word)
        if self.surface_form == "arabic":
            return str(self.arabic_word_map.get(word, word))
        return word

    def _sentence_surface(self, words: Sequence[str]) -> str:
        return " ".join(self._word_surface(str(word)) for word in words)

    def _format_slot(self, slot_name: str, entries: List[Dict[str, object]]) -> str:
        parts = []
        for item in entries:
            parts.append(
                f"{self._word_surface(str(item['word']))}:{float(item['probability']):.4f}"
            )
        return f"{slot_name}: " + ", ".join(parts)

    def _format_candidate(self, index: int, candidate: Dict[str, object]) -> str:
        slot_probs = ", ".join(
            f"{float(prob):.4f}" for prob in candidate.get("candidate_word_probabilities", [])
        )
        sentence_surface = self._sentence_surface(candidate.get("words", []))
        if self.prompt_language == "english":
            return (
                f"{index}. {candidate['sentence_id']} | {sentence_surface} | "
                f"slot_probs=({slot_probs}) | eeg_logprob={float(candidate['eeg_logprob']):.4f}"
            )
        return (
            f"{index}. {sentence_surface} | "
            f"احتمالات الكلمات=({slot_probs}) | درجة الرصد={float(candidate['eeg_logprob']):.4f}"
        )

    def build_prompt(
        self,
        slot_topk: List[List[Dict[str, object]]],
        retrieved_candidates: List[Dict[str, object]],
    ) -> str:
        if self.prompt_language == "english":
            slot_lines = [
                self._format_slot(f"Slot {idx + 1}", entries)
                for idx, entries in enumerate(slot_topk)
            ]
        else:
            slot_lines = [
                self._format_slot(f"الكلمة {idx + 1}", entries)
                for idx, entries in enumerate(slot_topk)
            ]
        candidate_lines = [
            self._format_candidate(idx + 1, candidate)
            for idx, candidate in enumerate(retrieved_candidates)
        ]

        if self.prompt_language == "english":
            return (
                "You are reranking candidate 3-word sentences from EEG evidence.\n"
                "Choose exactly one candidate.\n"
                "Use the slot probabilities as the main evidence.\n"
                "Prefer candidates whose words match the high-probability EEG words at each slot.\n"
                "Output only the integer index of the best candidate and nothing else.\n\n"
                "EEG slot evidence:\n"
                f"{chr(10).join(slot_lines)}\n\n"
                "Candidate shortlist:\n"
                f"{chr(10).join(candidate_lines)}\n\n"
                f"Return only one integer from 1 to {len(retrieved_candidates)}."
            )
        return (
            "أنت تعيد ترتيب جمل عربية مكونة من ثلاث كلمات اعتمادا على احتمالات الإشارة العصبية.\n"
            "اختر مرشحا واحدا فقط.\n"
            "اعتمد أساسا على احتمالات كل كلمة في كل موضع.\n"
            "فضّل الجملة التي تطابق الكلمات الأعلى احتمالا في المواضع الثلاثة.\n"
            "أعد رقم المرشح الأفضل فقط من دون أي شرح.\n\n"
            "أدلة الكلمات:\n"
            f"{chr(10).join(slot_lines)}\n\n"
            "المرشحون:\n"
            f"{chr(10).join(candidate_lines)}\n\n"
            f"أعد رقما واحدا فقط من 1 إلى {len(retrieved_candidates)}."
        )

    @staticmethod
    def _parse_choice(text: str, candidate_count: int) -> Optional[int]:
        for match in re.finditer(r"\d+", str(text)):
            normalized_digits: List[str] = []
            for char in match.group(0):
                if char.isascii() and char.isdigit():
                    normalized_digits.append(char)
                    continue
                try:
                    normalized_digits.append(str(unicodedata.digit(char)))
                except (TypeError, ValueError):
                    normalized_digits = []
                    break
            if not normalized_digits:
                continue
            choice = int("".join(normalized_digits))
            if 1 <= choice <= int(candidate_count):
                return choice
        return None

    def _build_generation_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        self._load_model()
        if self.prompt_language == "english":
            system_prompt = (
                "You are a careful sentence reranker. Follow the user's instructions exactly."
            )
        else:
            system_prompt = "أنت تعيد ترتيب الجمل بعناية. اتبع التعليمات بدقة."
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{system_prompt}\n\n{prompt}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        device = self._input_device()
        return {k: v.to(device) for k, v in inputs.items()}

    @torch.no_grad()
    def _generate_raw_output(self, prompt: str) -> str:
        inputs = self._build_generation_inputs(prompt)
        prompt_length = int(inputs["input_ids"].shape[1])
        generation = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = generation[0][prompt_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return str(text).strip()

    def select_candidate(
        self,
        slot_topk: List[List[Dict[str, object]]],
        retrieved_candidates: List[Dict[str, object]],
    ) -> Dict[str, object]:
        if not retrieved_candidates:
            raise ValueError("The LLM reranker requires at least one retrieved candidate.")

        prompt = self.build_prompt(slot_topk, retrieved_candidates)
        cache_key = hash_text(f"{self.model_name}\n{prompt}")
        cached = self.cache.get(cache_key)
        if cached is not None:
            raw_output = str(cached.get("raw_output", ""))
            parsed_choice = self._parse_choice(raw_output, len(retrieved_candidates))
            if parsed_choice is None:
                cached_choice = cached.get("parsed_choice")
                parsed_choice = int(cached_choice) if cached_choice is not None else None
            used_fallback = bool(cached.get("used_fallback", parsed_choice is None))
            fallback_reason = str(cached.get("fallback_reason", "cache"))
            return {
                "prompt": prompt,
                "prompt_hash": cache_key,
                "raw_output": raw_output,
                "parsed_choice": parsed_choice,
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "selected_index": int(parsed_choice or 1),
                "cache_hit": True,
            }

        raw_output = self._generate_raw_output(prompt)
        parsed_choice = self._parse_choice(raw_output, len(retrieved_candidates))
        used_fallback = parsed_choice is None
        fallback_reason = "invalid_or_missing_integer" if used_fallback else ""
        selected_index = int(parsed_choice or 1)

        payload = {
            "raw_output": raw_output,
            "parsed_choice": parsed_choice,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
        }
        self.cache.set(cache_key, payload)

        return {
            "prompt": prompt,
            "prompt_hash": cache_key,
            "raw_output": raw_output,
            "parsed_choice": parsed_choice,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "selected_index": selected_index,
            "cache_hit": False,
        }


class LLMRAGSentenceDecoder:
    def __init__(
        self,
        label_order: Sequence[str],
        sentence_catalog: List[Dict[str, object]],
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        hf_timeout: int = 120,
        hf_retries: int = 3,
        embedding_local_only: bool = False,
        llm_local_only: bool = False,
        embedding_cache_path: Optional[str] = None,
        llm_response_cache_path: Optional[str] = None,
        llm_max_new_tokens: int = 8,
        arabic_word_map: Optional[Dict[str, str]] = None,
        prompt_language: str = "arabic",
        surface_form: str = "arabic",
    ) -> None:
        self.retriever = ProbabilitySentenceRetriever(
            embedding_model_name=embedding_model_name,
            label_order=label_order,
            sentence_catalog=sentence_catalog,
            device=device,
            hf_timeout=hf_timeout,
            hf_retries=hf_retries,
            local_only=embedding_local_only,
            cache_path=embedding_cache_path,
        )
        self.reranker = LLMCandidateReranker(
            model_name=llm_model_name,
            device=device,
            hf_timeout=hf_timeout,
            hf_retries=hf_retries,
            local_only=llm_local_only,
            cache_path=llm_response_cache_path,
            max_new_tokens=llm_max_new_tokens,
            arabic_word_map=arabic_word_map,
            prompt_language=prompt_language,
            surface_form=surface_form,
        )

    def decode_trial(
        self,
        position_prob_vectors: Sequence[np.ndarray],
        slot_topk: List[List[Dict[str, object]]],
        retrieval_topk: int = 5,
        retrieval_mode: str = "hybrid",
        hybrid_alpha: float = 0.5,
        hybrid_normalization: str = "zscore",
        true_sentence_id: Optional[str] = None,
    ) -> Dict[str, object]:
        retrieval_output = self.retriever.retrieve(
            position_prob_vectors=position_prob_vectors,
            top_k=retrieval_topk,
            retrieval_mode=retrieval_mode,
            hybrid_alpha=hybrid_alpha,
            hybrid_normalization=hybrid_normalization,
        )
        ranked_candidates = list(retrieval_output["ranked_candidates"])
        retrieved_candidates = list(retrieval_output["retrieved_candidates"])

        reranker_output = self.reranker.select_candidate(slot_topk, retrieved_candidates)
        selected_index = max(1, min(int(reranker_output["selected_index"]), len(retrieved_candidates)))
        selected_candidate = dict(retrieved_candidates[selected_index - 1])

        true_sentence_rank = None
        true_in_topk = False
        top1_sentence_id = ""
        top1_correct = False

        if ranked_candidates:
            top1_sentence_id = str(ranked_candidates[0]["sentence_id"])
            if true_sentence_id is not None:
                true_sentence_id = str(true_sentence_id)
                top1_correct = top1_sentence_id == true_sentence_id

        if true_sentence_id is not None:
            for candidate in ranked_candidates:
                if str(candidate["sentence_id"]) == str(true_sentence_id):
                    true_sentence_rank = int(candidate["retrieval_rank"])
                    true_in_topk = true_sentence_rank <= int(retrieval_topk)
                    break

        return {
            "predicted_sentence_id": str(selected_candidate["sentence_id"]),
            "predicted": list(selected_candidate["words"]),
            "selected_candidate": selected_candidate,
            "retrieved_candidates": retrieved_candidates,
            "retrieval": {
                "mode": str(retrieval_output["retrieval_mode"]),
                "top_k": int(retrieval_topk),
                "hybrid_alpha": float(retrieval_output["hybrid_alpha"]),
                "hybrid_normalization": str(retrieval_output["hybrid_normalization"]),
                "candidate_count": int(retrieval_output["candidate_count"]),
                "query_vector_norm": float(retrieval_output["query_vector_norm"]),
                "true_sentence_rank": true_sentence_rank,
                "true_in_topk": bool(true_in_topk),
                "top1_sentence_id": top1_sentence_id,
                "top1_correct": bool(top1_correct),
            },
            "reranker": {
                "model_name": self.reranker.model_name,
                "prompt_hash": str(reranker_output["prompt_hash"]),
                "raw_output": str(reranker_output["raw_output"]),
                "parsed_choice": reranker_output["parsed_choice"],
                "selected_index": int(selected_index),
                "used_fallback": bool(reranker_output["used_fallback"]),
                "fallback_reason": str(reranker_output["fallback_reason"]),
                "cache_hit": bool(reranker_output["cache_hit"]),
            },
        }

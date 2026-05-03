#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for the copied official Stage 2 LLM-RAG runs.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def normalize_cli_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    path_str = str(path_str).strip()
    if not path_str:
        return path_str

    match = re.match(r"^/([a-zA-Z])/(.*)$", path_str)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("\\", "/")
        return f"{drive}:/{rest}"

    match = re.match(r"^/([a-zA-Z]:/.*)$", path_str.replace("\\", "/"))
    if match:
        return match.group(1)

    return path_str


def subject_sort_key(subject: str) -> Tuple[int, str]:
    match = re.match(r"^[sS](\d+)$", str(subject).strip())
    if match:
        return int(match.group(1)), str(subject)
    return 10**9, str(subject)


def sentence_sort_key(sentence_id: str) -> Tuple[int, str]:
    match = re.match(r"^[cC](\d+)$", str(sentence_id).strip())
    if match:
        return int(match.group(1)), str(sentence_id)
    return 10**9, str(sentence_id)


def trial_sort_key(trial_tag: str) -> Tuple[int, str]:
    match = re.match(r"^[tT](\d+)$", str(trial_tag).strip())
    if match:
        return int(match.group(1)), str(trial_tag)
    return 10**9, str(trial_tag)


def discover_subjects(precomputed_root: Path) -> List[str]:
    subjects = [
        item.name
        for item in precomputed_root.iterdir()
        if item.is_dir() and item.name.upper().startswith("S")
    ]
    return sorted(subjects, key=subject_sort_key)


def load_session_map(path: str) -> Dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k).upper(): str(v).upper() for k, v in raw.items()}


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        if probs.size == 0:
            return probs
        return np.full((probs.size,), 1.0 / float(probs.size), dtype=np.float32)
    return (probs / total).astype(np.float32)


def ensure_suffix(path: Path, suffix: str) -> Path:
    if path.suffix.lower() == suffix.lower():
        return path
    return path.with_suffix(suffix)


class SentenceCorpus:
    """Minimal sentence corpus loader used by the copied official methods."""

    def __init__(self, structure_path: str, labels_path: str):
        with open(structure_path, "r", encoding="utf-8") as handle:
            self.structure = json.load(handle)
        with open(labels_path, "r", encoding="utf-8") as handle:
            self.word_labels = json.load(handle)

        self.idx_to_word = {int(value): str(key) for key, value in self.word_labels.items()}

        self.sentences: List[List[str]] = []
        self.sentence_ids: List[str] = []
        for sentence_id in sorted(self.structure, key=sentence_sort_key):
            row = self.structure[sentence_id]
            self.sentences.append([str(row["W1"]), str(row["W2"]), str(row["W3"])])
            self.sentence_ids.append(str(sentence_id).upper())

    def get_sentence(self, sentence_id: str) -> List[str]:
        return list(self.sentences[self.sentence_ids.index(str(sentence_id).upper())])

    def get_all_sentences(self) -> List[List[str]]:
        return [list(sentence) for sentence in self.sentences]


def summarize_stage2_json_dir(json_dir: Path) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for path in sorted(json_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        metrics = data.get("metrics", {})
        predictions = data.get("predictions", [])
        params = data.get("parameters", {})
        subject = str(data.get("parameters", {}).get("subject", path.stem))
        accuracy = metrics.get("accuracy", metrics.get("sentence_accuracy"))
        if accuracy is None:
            accuracy = (
                sum(1 for item in predictions if bool(item.get("correct"))) / float(len(predictions))
                if predictions else 0.0
            )
        wer = float(metrics.get("wer", metrics.get("word_error_rate", 0.0)))
        retrieval_topk = int(params.get("retrieval_topk", 5))
        rows.append(
            {
                "subject": subject,
                "accuracy": float(accuracy),
                "wer": wer,
                "retrieval_only_accuracy": float(metrics.get("retrieval_only_accuracy", 0.0)),
                "retrieval_at_k": float(metrics.get(f"retrieval_recall_at_{retrieval_topk}", 0.0)),
                "retrieval_mrr": float(metrics.get("retrieval_mrr", 0.0)),
                "fallback_rate": float(metrics.get("reranker_fallback_rate", 0.0)),
                "retrieval_topk": float(retrieval_topk),
            }
        )

    if not rows:
        return {
            "subject_count": 0,
            "mean_accuracy": 0.0,
            "mean_wer": 0.0,
            "mean_retrieval_only_accuracy": 0.0,
            "mean_retrieval_at_k": 0.0,
            "mean_retrieval_mrr": 0.0,
            "mean_fallback_rate": 0.0,
            "retrieval_topk": 0,
        }

    subject_count = len(rows)
    return {
        "subject_count": subject_count,
        "mean_accuracy": sum(float(item["accuracy"]) for item in rows) / float(subject_count),
        "mean_wer": sum(float(item["wer"]) for item in rows) / float(subject_count),
        "mean_retrieval_only_accuracy": (
            sum(float(item["retrieval_only_accuracy"]) for item in rows) / float(subject_count)
        ),
        "mean_retrieval_at_k": (
            sum(float(item["retrieval_at_k"]) for item in rows) / float(subject_count)
        ),
        "mean_retrieval_mrr": sum(float(item["retrieval_mrr"]) for item in rows) / float(subject_count),
        "mean_fallback_rate": sum(float(item["fallback_rate"]) for item in rows) / float(subject_count),
        "retrieval_topk": int(rows[0]["retrieval_topk"]),
    }


@dataclass
class PrecomputedTrialRecord:
    classifier: str
    sentence_id: str
    trial_tag: str
    word_position: str
    true_word: str
    true_class_idx: int
    predicted_word: str
    predicted_class_idx: int
    probabilities: np.ndarray
    fold: int
    trial_position: int
    trial_key: str
    trial_name: str


class PrecomputedSubjectWordScores:
    """Load and resolve precomputed DiFFE trial probability distributions."""

    def __init__(
        self,
        subject: str,
        data_root: str,
        precomputed_root: str,
        session_map_path: Optional[str] = None,
        default_classifier: str = "A",
    ) -> None:
        self.subject = str(subject)
        self.data_root = str(data_root)
        self.precomputed_root = Path(precomputed_root)
        self.subject_dir = self.precomputed_root / self.subject
        if not self.subject_dir.exists():
            raise FileNotFoundError(f"Missing precomputed subject directory: {self.subject_dir}")

        sentence_structure_path = Path(self.data_root) / "sentence_structure.json"
        labels_path = Path(self.data_root) / "labels.json"
        self.corpus = SentenceCorpus(str(sentence_structure_path), str(labels_path))

        with open(labels_path, "r", encoding="utf-8") as handle:
            self.label_to_idx = {str(k): int(v) for k, v in json.load(handle).items()}

        if session_map_path is None:
            session_map_path = str(Path(self.data_root) / "session_map.json")
        self.session_map_path = str(session_map_path)
        self.session_map = load_session_map(self.session_map_path)
        self.default_classifier = str(default_classifier).upper()

        self.class_labels: List[str] = []
        self.idx_to_label: Dict[int, str] = {}
        self.records_by_classifier: Dict[str, Dict[Tuple[str, str, str], PrecomputedTrialRecord]] = {
            "A": {},
            "B": {},
        }
        self.trial_tags_by_sentence: Dict[str, set] = {sid: set() for sid in self.corpus.sentence_ids}
        self.source_paths: Dict[str, str] = {}

        for classifier in ("A", "B"):
            self._load_classifier_payload(classifier)

    def _load_classifier_payload(self, classifier: str) -> None:
        json_path = self.subject_dir / f"classifier_{classifier}_results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing precomputed classifier JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        class_labels = [str(item) for item in payload.get("class_labels", [])]
        if not class_labels:
            raise ValueError(f"No class labels found in {json_path}")

        if not self.class_labels:
            self.class_labels = class_labels
            self.idx_to_label = {idx: label for idx, label in enumerate(class_labels)}
        elif self.class_labels != class_labels:
            raise ValueError(
                f"Class label order mismatch between classifier payloads for {self.subject}: "
                f"{json_path}"
            )

        missing_labels = [label for label in self.class_labels if label not in self.label_to_idx]
        if missing_labels:
            raise ValueError(
                f"labels.json is missing {len(missing_labels)} labels required by {json_path}: "
                f"{missing_labels[:5]}"
            )

        record_map = self.records_by_classifier[classifier]
        for raw_record in payload.get("trial_probability_distributions", []):
            sentence_id = str(raw_record["session"]).upper()
            trial_tag = str(raw_record["trial_tag"]).upper()
            true_word = str(raw_record["true_class_label"])
            word_position = self._infer_word_position(sentence_id, true_word)
            probabilities = normalize_probs(np.asarray(raw_record["probabilities"], dtype=np.float32))
            trial_name = (
                f"{self.subject}_{sentence_id}_{trial_tag}_{word_position}_{true_word}"
            )

            record = PrecomputedTrialRecord(
                classifier=classifier,
                sentence_id=sentence_id,
                trial_tag=trial_tag,
                word_position=word_position,
                true_word=true_word,
                true_class_idx=int(raw_record["true_class_idx"]),
                predicted_word=str(raw_record["predicted_class_label"]),
                predicted_class_idx=int(raw_record["predicted_class_idx"]),
                probabilities=probabilities,
                fold=int(raw_record["fold"]),
                trial_position=int(raw_record["trial_position"]),
                trial_key=str(raw_record["trial_key"]),
                trial_name=trial_name,
            )

            key = (sentence_id, trial_tag, word_position)
            if key in record_map:
                raise ValueError(
                    f"Duplicate precomputed record for {self.subject} classifier {classifier}: {key}"
                )
            record_map[key] = record
            self.trial_tags_by_sentence.setdefault(sentence_id, set()).add(trial_tag)

        self.source_paths[classifier] = str(json_path)

    def _infer_word_position(self, sentence_id: str, true_word: str) -> str:
        sentence_map = self.corpus.structure.get(sentence_id)
        if not sentence_map:
            raise KeyError(f"Sentence '{sentence_id}' not found in sentence_structure.json")

        matches = [
            word_position
            for word_position, word in sentence_map.items()
            if str(word) == str(true_word)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not infer unique word position for {self.subject} {sentence_id} "
                f"word '{true_word}'. Matches: {matches}"
            )
        return str(matches[0]).upper()

    def preferred_classifier(self, sentence_id: str, word_position: str) -> str:
        key_pos = f"{sentence_id.upper()}_{word_position.upper()}"
        if key_pos in self.session_map:
            return self.session_map[key_pos]
        return self.session_map.get(sentence_id.upper(), self.default_classifier)

    def memberships(self, sentence_id: str, trial_tag: str, word_position: str) -> List[str]:
        key = (sentence_id.upper(), trial_tag.upper(), word_position.upper())
        return [
            classifier
            for classifier in ("A", "B")
            if key in self.records_by_classifier[classifier]
        ]

    def available_trial_tags(self, sentence_id: str) -> List[str]:
        return sorted(self.trial_tags_by_sentence.get(sentence_id.upper(), set()), key=trial_sort_key)

    def sentence_trial_counts(self) -> Dict[str, int]:
        return {
            sentence_id: len(self.available_trial_tags(sentence_id))
            for sentence_id in sorted(self.corpus.sentence_ids, key=sentence_sort_key)
        }

    def validate_coverage(self) -> List[Dict[str, object]]:
        issues: List[Dict[str, object]] = []
        for sentence_id in sorted(self.corpus.sentence_ids, key=sentence_sort_key):
            trial_tags = self.available_trial_tags(sentence_id)
            if not trial_tags:
                issues.append(
                    {
                        "sentence_id": sentence_id,
                        "word_position": "*",
                        "reason": "missing_sentence_trials",
                        "trial_tags": [],
                    }
                )
                continue

            for trial_tag in trial_tags:
                for word_position in ("W1", "W2", "W3"):
                    memberships = self.memberships(sentence_id, trial_tag, word_position)
                    if memberships:
                        continue
                    issues.append(
                        {
                            "sentence_id": sentence_id,
                            "trial_tag": trial_tag,
                            "word_position": word_position,
                            "reason": "no_precomputed_probability_vector",
                            "preferred_classifier": self.preferred_classifier(sentence_id, word_position),
                        }
                    )
        return issues

    def resolve_record(
        self,
        sentence_id: str,
        trial_tag: str,
        word_position: str,
    ) -> Tuple[PrecomputedTrialRecord, str]:
        sentence_id = sentence_id.upper()
        trial_tag = trial_tag.upper()
        word_position = word_position.upper()
        preferred = self.preferred_classifier(sentence_id, word_position)
        memberships = self.memberships(sentence_id, trial_tag, word_position)

        if preferred in memberships:
            selection = (
                "preferred_classifier_match_shared_trial"
                if len(memberships) > 1 else
                "preferred_classifier_match"
            )
            return self.records_by_classifier[preferred][(sentence_id, trial_tag, word_position)], selection

        if len(memberships) == 1:
            classifier = memberships[0]
            return (
                self.records_by_classifier[classifier][(sentence_id, trial_tag, word_position)],
                "auto_switched_to_trial_pool",
            )

        raise KeyError(
            f"Missing precomputed record for {self.subject} {sentence_id} {trial_tag} {word_position}"
        )

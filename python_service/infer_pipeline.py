from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .features import build_stage1_tensor
from .infer_stage1 import infer_stage1_posteriors
from .load_checkpoint import load_checkpoint
from .logging_utils import append_log
from .preprocess import preprocess_edf
from .rag_decoder import rerank_with_qwen
from .retrieve_sentences import retrieve_candidates
from .segment import segment_imagination_windows
from .storage import (
    ensure_profile_layout,
    now_iso,
    profile_inference_dir,
    read_json,
    write_json,
)
from .config import LABELS_PATH


def _label_ids() -> list[int]:
    payload = read_json(LABELS_PATH, {"labels": []})
    return [int(item["id"]) for item in payload.get("labels", [])]


def _session_id(prefix: str) -> str:
    return f"{prefix}-{int(np.datetime64('now', 'ms').astype(int))}"


def run_inference_pipeline(profile_id: str, user_model_path: str, edf_path: str, simulated: bool = False) -> Dict[str, object]:
    ensure_profile_layout(profile_id)
    session_id = _session_id("infer")

    load_checkpoint(user_model_path)

    preprocessed = preprocess_edf(profile_id, edf_path)
    segmented = segment_imagination_windows(profile_id, preprocessed.data, preprocessed.sfreq, Path(edf_path).stem)

    slot_tensors = []
    band_power_summary = []
    for window in segmented.windows:
        tensor, powers = build_stage1_tensor(window, preprocessed.sfreq)
        slot_tensors.append(tensor)
        band_power_summary.append(powers)

    stage1_posteriors = infer_stage1_posteriors(slot_tensors, _label_ids())
    candidates = retrieve_candidates(stage1_posteriors)
    selected, used_fallback, llm_raw = rerank_with_qwen(candidates, stage1_posteriors)

    final_sentence = str(selected["arabic"])
    selected_sentence_id = int(selected["sentence_id"])

    inference_dir = profile_inference_dir(profile_id)
    manifest_path = inference_dir / "inference_manifest.json"
    result_path = inference_dir / "results" / f"{session_id}.json"

    payload = {
        "session_id": session_id,
        "profile_id": profile_id,
        "created_at": now_iso(),
        "simulated": simulated,
        "source_edf": edf_path,
        "preprocessed_path": str(preprocessed.output_path),
        "segmented_windows": [str(path) for path in segmented.window_paths],
        "stage1_posteriors": stage1_posteriors,
        "retrieval_candidates": candidates,
        "selected_sentence_id": selected_sentence_id,
        "final_sentence": final_sentence,
        "used_fallback": used_fallback,
        "llm_raw": llm_raw,
        "band_power_summary": band_power_summary,
        "status": "success",
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, payload)
    write_json(manifest_path, payload)

    append_log("infer.log", f"[{profile_id}] inference complete -> sentence_id={selected_sentence_id}")

    return {
        "status": "success",
        "message": "Session inference complete.",
        "session_id": session_id,
        "final_sentence": final_sentence,
        "selected_sentence_id": selected_sentence_id,
        "stage1_posteriors": stage1_posteriors,
        "candidates": [
            {
                "sentence_id": int(item["sentence_id"]),
                "arabic": item["arabic"],
                "score": float(item["score"]),
            }
            for item in candidates
        ],
        "used_fallback": used_fallback,
    }


def run_simulated_pipeline(profile_id: str, user_model_path: Optional[str], model_ready: bool) -> Dict[str, object]:
    points = []
    channels = [
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
    ]

    for t in range(120):
        values = [float(np.sin((t + idx * 5) / 9.0) * 22 + np.cos((t + idx) / 7.0) * 4) for idx in range(14)]
        points.append({"t": t, "values": values})

    simulation = {
        "signal_status": "complete",
        "current_step": "Simulated replay complete",
        "channel_names": channels,
        "points": points,
    }

    inference_result = None
    if model_ready and user_model_path:
        slot_probs = infer_stage1_posteriors([np.ones((19, 1400))] * 3, _label_ids())
        candidates = retrieve_candidates(slot_probs)
        selected, used_fallback, _ = rerank_with_qwen(candidates, slot_probs)
        inference_result = {
            "status": "success",
            "message": "Simulation produced a sentence preview.",
            "session_id": _session_id("sim"),
            "final_sentence": selected["arabic"],
            "selected_sentence_id": selected["sentence_id"],
            "stage1_posteriors": slot_probs,
            "candidates": [
                {
                    "sentence_id": int(item["sentence_id"]),
                    "arabic": item["arabic"],
                    "score": float(item["score"]),
                }
                for item in candidates
            ],
            "used_fallback": used_fallback,
        }

    return {
        "status": "success",
        "message": "Simulation completed.",
        "simulation": simulation,
        "inference_result": inference_result,
    }

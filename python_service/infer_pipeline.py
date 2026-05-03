from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import DEMO_EDF_PATH, DEMO_MARKER_PATH, DEMO_MODEL_PATH
from .edf_trial_processor import process_trial_edf
from .logging_utils import append_log
from .stage1_model_adapter import Stage1DiffEAdapter
from .stage2_decoder_adapter import Stage2DecoderAdapter
from .storage import ensure_profile_layout, now_iso, profile_inference_dir, write_json


def _session_id(prefix: str) -> str:
    return f"{prefix}-{int(np.datetime64('now', 'ms').astype(int))}"


def _new_timeline() -> List[Dict[str, object]]:
    return [
        {"id": "model_validated", "label": "Model validated", "status": "pending", "detail": None, "warnings": []},
        {"id": "edf_loaded", "label": "EDF loaded", "status": "pending", "detail": None, "warnings": []},
        {"id": "marker_detected", "label": "Marker CSV detected", "status": "pending", "detail": None, "warnings": []},
        {"id": "preprocessing", "label": "Preprocessing EEG", "status": "pending", "detail": None, "warnings": []},
        {
            "id": "segmenting",
            "label": "Segmenting 3 imagination windows",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
        {
            "id": "building_tensors",
            "label": "Building 19-channel tensors",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
        {
            "id": "stage1",
            "label": "Running Stage 1 DiffE word classifier",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
        {
            "id": "posterior_evidence",
            "label": "Building Top-k posterior evidence",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
        {
            "id": "retrieval",
            "label": "Building transformer candidate shortlist",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
        {
            "id": "reranking",
            "label": "Running Qwen/Ollama sentence selection",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
        {
            "id": "decoded",
            "label": "Final sentence decoded",
            "status": "pending",
            "detail": None,
            "warnings": [],
        },
    ]


def _set_step(
    timeline: List[Dict[str, object]],
    step_id: str,
    status: str,
    detail: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> None:
    for step in timeline:
        if step["id"] != step_id:
            continue
        step["status"] = status
        if detail is not None:
            step["detail"] = detail
        if warnings:
            step["warnings"] = list(warnings)
            if status == "complete":
                step["status"] = "warning"
        return


def _failure_payload(profile_id: str, session_id: str, message: str, timeline: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "status": "failed",
        "message": message,
        "session_id": session_id,
        "profile_id": profile_id,
        "model": {"validated": False},
        "edf": {},
        "preprocessing": {"warnings": []},
        "stage1": {"top_k_words": 0, "slots": []},
        "stage2": {
            "mode": "qwen",
            "retrieval_topk": 0,
            "used_fallback": True,
            "candidate_sentences": [],
            "warnings": [],
        },
        "prediction": {},
        "timing": {"total_ms": 0, "preprocessing_ms": 0, "stage1_ms": 0, "stage2_ms": 0},
        "timeline": timeline,
        "final_sentence": None,
        "selected_sentence_id": None,
        "used_fallback": True,
    }


def run_inference_pipeline(
    profile_id: str,
    user_model_path: str,
    edf_path: str,
    marker_csv_path: Optional[str] = None,
    top_k_words: int = 8,
    retrieval_topk: int = 5,
    stage2_mode: str = "qwen",
    use_demo_files: bool = False,
) -> Dict[str, object]:
    ensure_profile_layout(profile_id)
    session_id = _session_id("infer")
    timeline = _new_timeline()
    started = time.perf_counter()

    resolved_model_path = str(DEMO_MODEL_PATH if use_demo_files else Path(user_model_path))
    resolved_edf_path = str(DEMO_EDF_PATH if use_demo_files else Path(edf_path))
    resolved_marker_path = str(DEMO_MARKER_PATH) if use_demo_files else marker_csv_path

    if not resolved_model_path:
        raise ValueError("A personalized Stage 1 checkpoint is required for inference.")
    if not resolved_edf_path:
        raise ValueError("edf_path is required for inference.")

    try:
        model = Stage1DiffEAdapter(model_path=resolved_model_path)
        _set_step(timeline, "model_validated", "complete", detail=model.info()["filename"])

        _set_step(timeline, "edf_loaded", "running")
        preprocessing_started = time.perf_counter()
        processed_trial = process_trial_edf(edf_path=resolved_edf_path, marker_csv_path=resolved_marker_path)
        preprocessing_ms = int((time.perf_counter() - preprocessing_started) * 1000)
        _set_step(timeline, "edf_loaded", "complete", detail=Path(processed_trial.edf_path).name)
        _set_step(timeline, "marker_detected", "complete", detail=Path(processed_trial.marker_csv_path).name)
        _set_step(
            timeline,
            "preprocessing",
            "complete",
            detail=f"{processed_trial.sampling_rate} Hz, {len(processed_trial.channels)} EEG channels",
            warnings=processed_trial.warnings,
        )
        _set_step(
            timeline,
            "segmenting",
            "complete",
            detail=f"{len(processed_trial.imagine_markers)} imagination windows",
        )
        _set_step(
            timeline,
            "building_tensors",
            "complete",
            detail=", ".join([f"{shape[0]}x{shape[1]}" for shape in processed_trial.slot_tensor_shapes]),
        )

        _set_step(timeline, "stage1", "running")
        stage1_started = time.perf_counter()
        stage1_slots = model.predict_slots(processed_trial.slot_tensors, top_k=top_k_words)
        stage1_ms = int((time.perf_counter() - stage1_started) * 1000)
        _set_step(timeline, "stage1", "complete", detail=f"{len(stage1_slots)} slots classified")
        _set_step(timeline, "posterior_evidence", "complete", detail=f"Top-{int(top_k_words)} evidence prepared")

        _set_step(timeline, "retrieval", "running")
        stage2_started = time.perf_counter()
        decoder = Stage2DecoderAdapter()
        stage2_output = decoder.decode(stage1_slots=stage1_slots, retrieval_topk=retrieval_topk, stage2_mode=stage2_mode)
        stage2_ms = int((time.perf_counter() - stage2_started) * 1000)
        _set_step(
            timeline,
            "retrieval",
            "complete",
            detail=f"{len(stage2_output['candidate_sentences'])} transformer-ranked candidates returned",
        )
        _set_step(
            timeline,
            "reranking",
            "complete",
            detail="Retrieval rank-1 fallback used" if stage2_output["used_fallback"] else "Qwen/Ollama selected a candidate",
            warnings=stage2_output["warnings"],
        )
        _set_step(
            timeline,
            "decoded",
            "complete",
            detail=f"{stage2_output['prediction']['sentence_id']} selected",
        )

        total_ms = int((time.perf_counter() - started) * 1000)
        payload = {
            "status": "success",
            "message": "Khayal inference complete.",
            "session_id": session_id,
            "profile_id": profile_id,
            "model": model.info(),
            "edf": {
                "path": processed_trial.edf_path,
                "filename": Path(processed_trial.edf_path).name,
                "marker_csv": processed_trial.marker_csv_path,
                "subject": processed_trial.subject,
                "trial": processed_trial.trial,
                "sentence_id": processed_trial.sentence_id,
            },
            "preprocessing": {
                "channels": processed_trial.channels,
                "sampling_rate": processed_trial.sampling_rate,
                "num_slots": len(processed_trial.slot_tensors),
                "slot_tensor_shapes": processed_trial.slot_tensor_shapes,
                "marker_csv": processed_trial.marker_csv_path,
                "warnings": processed_trial.warnings,
                "imagine_markers": processed_trial.imagine_markers,
            },
            "stage1": {
                "top_k_words": int(top_k_words),
                "slots": stage1_slots,
            },
            "stage2": {
                "mode": stage2_output["mode"],
                "retrieval_topk": stage2_output["retrieval_topk"],
                "used_fallback": stage2_output["used_fallback"],
                "candidate_sentences": stage2_output["candidate_sentences"],
                "raw_llm_output": stage2_output["raw_llm_output"],
                "warnings": stage2_output["warnings"],
                "reranker_model": stage2_output["reranker_model"],
                "transformer_model": stage2_output["transformer_model"],
                "device": stage2_output["device"],
                "transformer_retrieval_used": stage2_output["transformer_retrieval_used"],
            },
            "prediction": stage2_output["prediction"],
            "timing": {
                "total_ms": total_ms,
                "preprocessing_ms": preprocessing_ms,
                "stage1_ms": stage1_ms,
                "stage2_ms": stage2_ms,
            },
            "timeline": timeline,
            "final_sentence": stage2_output["prediction"]["arabic"],
            "selected_sentence_id": stage2_output["prediction"]["sentence_id"],
            "used_fallback": stage2_output["used_fallback"],
            "created_at": now_iso(),
        }

        inference_dir = profile_inference_dir(profile_id)
        result_path = inference_dir / "results" / f"{session_id}.json"
        manifest_path = inference_dir / "inference_manifest.json"
        write_json(result_path, payload)
        write_json(manifest_path, payload)
        append_log(
            "infer.log",
            f"[{profile_id}] inference complete -> sentence_id={payload['prediction']['sentence_id']}",
        )
        return payload
    except Exception as exc:
        total_ms = int((time.perf_counter() - started) * 1000)
        _set_step(timeline, "decoded", "failed", detail=str(exc))
        payload = _failure_payload(profile_id=profile_id, session_id=session_id, message=str(exc), timeline=timeline)
        payload["timing"] = {
            "total_ms": total_ms,
            "preprocessing_ms": 0,
            "stage1_ms": 0,
            "stage2_ms": 0,
        }
        append_log("infer.log", f"[{profile_id}] inference failed: {exc}")
        return payload


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

    return {
        "status": "success",
        "message": "Simulated-only replay completed.",
        "simulation": {
            "signal_status": "complete",
            "current_step": "Simulated-only replay complete",
            "channel_names": channels,
            "points": points,
        },
        "inference_result": None,
    }

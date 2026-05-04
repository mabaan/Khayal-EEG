from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import OLLAMA_BASE_URL, STAGE2_SUPPORTED_MODES
from .infer_pipeline import run_inference_pipeline, run_simulated_pipeline
from .logging_utils import append_log
from .schemas import (
    HealthResponse,
    InferRequest,
    InferResponse,
    SimulateRequest,
    SimulateResponse,
    TrainRequest,
    TrainResponse,
    ValidateModelRequest,
    ValidateModelResponse,
)
from .stage1_model_adapter import validate_checkpoint
from .stage2_decoder_adapter import cuda_available, ollama_available, transformers_available
from .storage import ensure_storage_layout
from .train_user_model import train_user_model

app = FastAPI(title="Khayal Local Python Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_storage_layout()
    append_log("app.log", "python_service startup complete")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok = ollama_available()
    transformers_ok = transformers_available()
    cuda_ok = cuda_available()
    status = "ok"
    message = f"Service online. Stage2 endpoint: {OLLAMA_BASE_URL}"
    if not ollama_ok:
        message += " Qwen/Ollama unavailable; retrieval rank-1 fallback is ready."
    if not cuda_ok:
        message += " CUDA unavailable; local retrieval will run on CPU."
    return HealthResponse(
        status=status,
        message=message,
        ollama_available=ollama_ok,
        transformers_available=transformers_ok,
        cuda_available=cuda_ok,
        supported_stage2_modes=list(STAGE2_SUPPORTED_MODES),
    )


@app.post("/validate-model", response_model=ValidateModelResponse)
def validate_model(request: ValidateModelRequest) -> ValidateModelResponse:
    try:
        metadata = validate_checkpoint(request.model_path)
        model = {
            "path": metadata["path"],
            "filename": metadata["filename"],
            "arch": metadata["arch"],
            "subject": metadata["subject"],
            "classifier": metadata["classifier"],
            "n_classes": metadata["n_classes"],
            "device": metadata["device"],
            "window_size": metadata["window_size"],
            "validated": True,
        }
        return ValidateModelResponse(
            status="success",
            message="Checkpoint validated successfully.",
            model=model,
        )
    except Exception as exc:
        return ValidateModelResponse(status="failed", message=str(exc), model=None)


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    try:
        result = train_user_model(
            profile_id=request.profile_id,
            base_model_path=request.base_model_path,
            calibration_edf_paths=request.calibration_edf_paths,
        )
        return TrainResponse(
            status="success",
            message=result["message"],
            profile_id=request.profile_id,
            session_id=result["session_id"],
            model_path=result["model_path"],
            metrics_path=result["metrics_path"],
        )
    except Exception as exc:
        append_log("train.log", f"[{request.profile_id}] training failed: {exc}")
        return TrainResponse(
            status="failed",
            message=str(exc),
            profile_id=request.profile_id,
            session_id="",
            model_path=None,
            metrics_path=None,
        )


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    result = run_inference_pipeline(
        profile_id=request.profile_id,
        user_model_path=request.user_model_path,
        edf_path=request.edf_path,
        marker_csv_path=request.marker_csv_path,
        top_k_words=request.top_k_words,
        retrieval_topk=request.retrieval_topk,
        stage2_mode=request.stage2_mode,
    )
    return InferResponse(**result)


@app.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> SimulateResponse:
    try:
        payload = run_simulated_pipeline(
            profile_id=request.profile_id,
            user_model_path=request.user_model_path,
            model_ready=request.model_ready,
        )
        return SimulateResponse(**payload)
    except Exception as exc:
        append_log("infer.log", f"[{request.profile_id}] simulation failed: {exc}")
        return SimulateResponse(
            status="failed",
            message=str(exc),
            simulation={
                "signal_status": "error",
                "current_step": "Simulation failed",
                "channel_names": [],
                "points": [],
            },
            inference_result=None,
        )

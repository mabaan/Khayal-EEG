from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import OLLAMA_BASE_URL
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
)
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
    return HealthResponse(status="ok", message=f"Service online. Stage2 endpoint: {OLLAMA_BASE_URL}")


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
    try:
        result = run_inference_pipeline(
            profile_id=request.profile_id,
            user_model_path=request.user_model_path,
            edf_path=request.edf_path,
            simulated=request.simulated,
        )
        return InferResponse(**result)
    except Exception as exc:
        append_log("infer.log", f"[{request.profile_id}] inference failed: {exc}")
        return InferResponse(
            status="failed",
            message=str(exc),
            session_id="",
            final_sentence=None,
            selected_sentence_id=None,
            stage1_posteriors=[],
            candidates=[],
            used_fallback=True,
        )


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

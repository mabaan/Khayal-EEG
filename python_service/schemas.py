from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    message: str


class TrainRequest(BaseModel):
    profile_id: str
    base_model_path: str
    calibration_edf_paths: List[str] = Field(default_factory=list)


class TrainResponse(BaseModel):
    status: Literal["success", "failed"]
    message: str
    profile_id: str
    session_id: str
    model_path: Optional[str] = None
    metrics_path: Optional[str] = None


class InferRequest(BaseModel):
    profile_id: str
    user_model_path: str
    edf_path: str
    simulated: bool = False


class WordPosterior(BaseModel):
    slot: int
    probabilities: Dict[str, float]


class RetrievalCandidate(BaseModel):
    sentence_id: int
    arabic: str
    score: float


class InferResponse(BaseModel):
    status: Literal["success", "failed"]
    message: str
    session_id: str
    final_sentence: Optional[str] = None
    selected_sentence_id: Optional[int] = None
    stage1_posteriors: List[WordPosterior] = Field(default_factory=list)
    candidates: List[RetrievalCandidate] = Field(default_factory=list)
    used_fallback: bool = False


class SimulateRequest(BaseModel):
    profile_id: str
    user_model_path: Optional[str] = None
    model_ready: bool = False


class SimPoint(BaseModel):
    t: int
    values: List[float]


class SimulationSnapshot(BaseModel):
    signal_status: str
    current_step: str
    channel_names: List[str]
    points: List[SimPoint]


class SimulateResponse(BaseModel):
    status: Literal["success", "failed"]
    message: str
    simulation: SimulationSnapshot
    inference_result: Optional[InferResponse] = None

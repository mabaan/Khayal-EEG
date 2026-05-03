from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    message: str
    ollama_available: bool = False
    transformers_available: bool = False
    cuda_available: bool = False
    supported_stage2_modes: List[str] = Field(default_factory=list)


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
    marker_csv_path: Optional[str] = None
    top_k_words: int = 8
    retrieval_topk: int = 5
    stage2_mode: Literal["qwen"] = "qwen"
    use_demo_files: bool = False


class TimelineStep(BaseModel):
    id: str
    label: str
    status: Literal["pending", "running", "complete", "warning", "failed"]
    detail: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    path: str = ""
    filename: str = ""
    arch: str = ""
    subject: Optional[str] = None
    classifier: Optional[str] = None
    n_classes: int = 0
    device: str = ""
    window_size: int = 0
    validated: bool = False


class EdfInfo(BaseModel):
    path: str = ""
    filename: str = ""
    marker_csv: Optional[str] = None
    subject: Optional[str] = None
    trial: Optional[str] = None
    sentence_id: Optional[str] = None


class Stage1TopWord(BaseModel):
    label_id: int
    word: str
    arabic: str
    probability: float


class Stage1SlotResult(BaseModel):
    slot: int
    probabilities: Dict[str, float] = Field(default_factory=dict)
    top_k: List[Stage1TopWord] = Field(default_factory=list)


class PreprocessingInfo(BaseModel):
    channels: List[str] = Field(default_factory=list)
    sampling_rate: int = 0
    num_slots: int = 0
    slot_tensor_shapes: List[List[int]] = Field(default_factory=list)
    marker_csv: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    imagine_markers: List[Dict[str, object]] = Field(default_factory=list)


class Stage1Info(BaseModel):
    top_k_words: int = 0
    slots: List[Stage1SlotResult] = Field(default_factory=list)


class Stage2Candidate(BaseModel):
    rank: int
    sentence_id: str
    arabic: str
    romanized: str
    english: Optional[str] = None
    retrieval_score: float
    posterior_score: float
    transformer_score: Optional[float] = None
    rerank_selected: bool = False
    word_probabilities: List[float] = Field(default_factory=list)


class Stage2Info(BaseModel):
    mode: str = "qwen"
    retrieval_topk: int = 0
    used_fallback: bool = False
    candidate_sentences: List[Stage2Candidate] = Field(default_factory=list)
    raw_llm_output: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    reranker_model: Optional[str] = None
    transformer_model: Optional[str] = None
    device: Optional[str] = None
    transformer_retrieval_used: bool = False


class PredictionInfo(BaseModel):
    sentence_id: str = ""
    arabic: str = ""
    romanized: str = ""
    english: Optional[str] = None
    score: Optional[float] = None


class TimingInfo(BaseModel):
    total_ms: int = 0
    preprocessing_ms: int = 0
    stage1_ms: int = 0
    stage2_ms: int = 0


class InferResponse(BaseModel):
    status: Literal["success", "failed"]
    message: str
    session_id: str
    profile_id: str = ""
    model: ModelInfo = Field(default_factory=ModelInfo)
    edf: EdfInfo = Field(default_factory=EdfInfo)
    preprocessing: PreprocessingInfo = Field(default_factory=PreprocessingInfo)
    stage1: Stage1Info = Field(default_factory=Stage1Info)
    stage2: Stage2Info = Field(default_factory=Stage2Info)
    prediction: PredictionInfo = Field(default_factory=PredictionInfo)
    timing: TimingInfo = Field(default_factory=TimingInfo)
    timeline: List[TimelineStep] = Field(default_factory=list)
    final_sentence: Optional[str] = None
    selected_sentence_id: Optional[str] = None
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


class ValidateModelRequest(BaseModel):
    model_path: str


class ValidateModelResponse(BaseModel):
    status: Literal["success", "failed"]
    message: str
    model: Optional[ModelInfo] = None

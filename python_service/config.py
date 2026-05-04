from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"

BASE_MODELS_DIR = STORAGE_DIR / "base_models"
PROFILES_DIR = STORAGE_DIR / "profiles"
SESSIONS_DIR = STORAGE_DIR / "sessions"
LOGS_DIR = STORAGE_DIR / "logs"

SENTENCE_CATALOG_PATH = DATA_DIR / "sentence_catalog.json"
SENTENCE_STRUCTURE_PATH = DATA_DIR / "sentence_structure.json"
SESSION_MAP_PATH = DATA_DIR / "session_map.json"
LABELS_PATH = DATA_DIR / "labels.json"
APP_DEFAULTS_PATH = DATA_DIR / "app_defaults.json"

BASE_MODEL_DEFAULT_PATH = BASE_MODELS_DIR / "diff_e_base.pt"
STAGE2_CACHE_DIR = STORAGE_DIR / "stage2_cache"

SAMPLING_RATE_HZ = 256
NOTCH_HZ = 50.0
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 32.5

WORD_WINDOWS_SECONDS = [
    {"slot": 1, "imagination": (10.0, 16.0), "keep": (10.5, 16.0)},
    {"slot": 2, "imagination": (26.0, 32.0), "keep": (26.5, 32.0)},
    {"slot": 3, "imagination": (42.0, 48.0), "keep": (42.5, 48.0)},
]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 32.5),
}

EMOTIV_CHANNELS = [
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

CHANNEL_STD_FLAT_THRESHOLD = 1e-6
CHANNEL_STD_NOISY_THRESHOLD = 350.0

STAGE2_TOP_K = 5
STAGE2_SUPPORTED_MODES = ["qwen"]
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_HEALTH_ENDPOINT = "/api/tags"
OLLAMA_GENERATE_ENDPOINT = "/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
RERANK_TEMPERATURE = 0.0
STAGE2_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STAGE2_RETRIEVAL_MODE = "hybrid"
STAGE2_HYBRID_ALPHA = 0.1
STAGE2_HYBRID_NORMALIZATION = "zscore"

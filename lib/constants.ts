import defaults from "@/data/app_defaults.json";

export const APP_NAME = defaults.app_name;
export const APP_TITLE = "Khayal";
export const LOCAL_MODE_LABEL = "Local Offline Mode";

export const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL ??
  process.env.NEXT_PUBLIC_PYTHON_SERVICE_URL ??
  `http://127.0.0.1:${defaults.ports.python_service}`;

export const STORAGE_ROOT = defaults.paths.storage_root;
export const BASE_MODEL_DEFAULT_PATH = defaults.paths.base_model;

export const CALIBRATION_STARTER_REPETITIONS = defaults.calibration.starter_repetitions;
export const CALIBRATION_FULL_REPETITIONS = defaults.calibration.full_repetitions;

export const SAMPLING_RATE_HZ = defaults.protocol.sampling_rate_hz;
export const REST_SECONDS = defaults.protocol.rest_seconds;
export const STIMULUS_SECONDS = defaults.protocol.stimulus_seconds;
export const IMAGINATION_SECONDS = defaults.protocol.imagination_seconds;
export const DISCARD_SECONDS = defaults.protocol.discard_seconds;
export const KEEP_SECONDS = defaults.protocol.keep_seconds;
export const WORD_WINDOWS = defaults.protocol.word_windows;

export const READINESS_ERROR_MESSAGE =
  "Inference is blocked: active profile does not have a personalized Stage 1 checkpoint.";

export const STAGE1_MODEL_NAME = "Diff-E";
export const STAGE2_MODEL_NAME = "Retrieval + Qwen";

export const OLLAMA_BASE_URL = defaults.stage2.ollama_base_url;
export const OLLAMA_MODEL_NAME = defaults.stage2.ollama_model;
export const STAGE2_TOP_K = defaults.stage2.top_k;
export const STAGE2_DETERMINISTIC_TEMPERATURE = defaults.stage2.deterministic_temperature;
export const STAGE2_FALLBACK_STRATEGY = defaults.stage2.fallback_strategy;
export const STAGE2_ORDERED_SLOTS = 3;
export const STAGE2_USE_FULL_POSTERIORS = true;

export const SESSION_WIDGET_LABELS = {
  session: "Session",
  liveSignals: "Live Signals",
  currentStep: "Current Step",
  signalStatus: "Signal Status",
  predictedSentence: "Predicted Sentence",
  recentSessions: "Recent Sessions",
  modelReady: "Model Ready",
  uploadEegRecording: "Upload EEG Recording"
} as const;

export const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/setup", label: "Setup" },
  { href: "/calibration", label: "Calibration" },
  { href: "/training", label: "Training" },
  { href: "/inference", label: "Session" },
  { href: "/history", label: "Recent Sessions" },
  { href: "/settings", label: "Settings" }
] as const;

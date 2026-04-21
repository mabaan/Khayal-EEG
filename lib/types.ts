export type ProfileStatus = "needs_calibration" | "training" | "ready";

export type SessionType = "calibration" | "training" | "inference" | "simulation";

export type SignalStatus = "idle" | "uploading" | "preprocessing" | "segmenting" | "running" | "complete" | "error";

export interface SentenceCatalogItem {
  sentence_id: number;
  arabic: string;
  english: string;
  word_ids: [number, number, number];
  word_tokens: [string, string, string];
}

export interface LabelItem {
  id: number;
  token: string;
}

export interface ProfileManifest {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  base_model_path: string;
  user_model_path: string | null;
  status: ProfileStatus;
  model_ready: boolean;
  notes: string;
}

export interface ActiveProfileRef {
  profile_id: string | null;
}

export interface SessionRecord {
  session_id: string;
  profile_id: string;
  type: SessionType;
  created_at: string;
  status: "success" | "failed";
  signal_status: SignalStatus;
  source: "edf_upload" | "simulated";
  input_path?: string;
  output_path?: string;
  predicted_sentence?: string;
  details?: Record<string, unknown>;
}

export interface TrainingRequest {
  profile_id: string;
  calibration_edf_paths: string[];
}

export interface TrainingResult {
  profile_id: string;
  session_id: string;
  model_path: string;
  metrics_path: string;
  status: "success" | "failed";
  message: string;
}

export interface InferenceRequest {
  profile_id: string;
  edf_path: string;
  simulated: boolean;
}

export interface WordPosterior {
  slot: 1 | 2 | 3;
  probabilities: Record<string, number>;
}

export interface RetrievalCandidate {
  sentence_id: number;
  arabic: string;
  score: number;
}

export interface InferenceResult {
  profile_id: string;
  session_id: string;
  final_sentence: string;
  candidates: RetrievalCandidate[];
  selected_sentence_id: number;
  stage1_posteriors: WordPosterior[];
  used_fallback: boolean;
  status: "success" | "failed";
  message: string;
}

export interface SimulatedSignalPoint {
  t: number;
  values: number[];
}

export interface SimulationSnapshot {
  signal_status: SignalStatus;
  current_step: string;
  channel_names: string[];
  points: SimulatedSignalPoint[];
}

export type ProfileStatus = "needs_calibration" | "ready";

export type SessionType = "calibration" | "inference" | "simulation";

export type SignalStatus = "idle" | "uploading" | "preprocessing" | "segmenting" | "running" | "complete" | "error";

export type Stage2Mode = "qwen";

export type TimelineStatus = "pending" | "running" | "complete" | "warning" | "failed";

export interface SentenceCatalogItem {
  sentence_id: string;
  arabic: string;
  romanized: string;
  english: string;
  word_ids: [number, number, number];
  word_tokens: [string, string, string];
  word_arabic: [string, string, string];
}

export interface LabelItem {
  id: number;
  word: string;
  arabic: string;
  english: string;
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

export interface InferenceRequest {
  edf_path: string;
  marker_csv_path?: string;
  top_k_words?: number;
  retrieval_topk?: number;
  stage2_mode?: Stage2Mode;
}

export interface TimelineStep {
  id: string;
  label: string;
  status: TimelineStatus;
  detail?: string | null;
  warnings: string[];
}

export interface ModelInfo {
  path: string;
  filename: string;
  arch: string;
  subject?: string | null;
  classifier?: string | null;
  n_classes: number;
  device: string;
  window_size: number;
  validated: boolean;
}

export interface EdfInfo {
  path: string;
  filename: string;
  marker_csv?: string | null;
  subject?: string | null;
  trial?: string | null;
  sentence_id?: string | null;
}

export interface Stage1TopWord {
  label_id: number;
  word: string;
  arabic: string;
  probability: number;
}

export interface WordPosterior {
  slot: 1 | 2 | 3;
  probabilities: Record<string, number>;
  top_k?: Stage1TopWord[];
}

export interface PreprocessingInfo {
  channels: string[];
  sampling_rate: number;
  num_slots: number;
  slot_tensor_shapes: number[][];
  marker_csv?: string | null;
  warnings: string[];
  imagine_markers: Array<Record<string, unknown>>;
}

export interface Stage1Info {
  top_k_words: number;
  slots: WordPosterior[];
}

export interface Stage2Candidate {
  rank: number;
  sentence_id: string;
  arabic: string;
  romanized: string;
  english?: string | null;
  retrieval_score: number;
  posterior_score: number;
  transformer_score?: number | null;
  rerank_selected: boolean;
  word_probabilities: number[];
}

export interface Stage2Info {
  mode: Stage2Mode;
  retrieval_topk: number;
  used_fallback: boolean;
  candidate_sentences: Stage2Candidate[];
  raw_llm_output?: string | null;
  warnings: string[];
  reranker_model?: string | null;
  transformer_model?: string | null;
  device?: string | null;
  transformer_retrieval_used: boolean;
}

export interface PredictionInfo {
  sentence_id: string;
  arabic: string;
  romanized: string;
  english?: string | null;
  score?: number | null;
}

export interface TimingInfo {
  total_ms: number;
  preprocessing_ms: number;
  stage1_ms: number;
  stage2_ms: number;
}

export interface InferenceResult {
  profile_id: string;
  session_id: string;
  status: "success" | "failed";
  message: string;
  model: ModelInfo;
  edf: EdfInfo;
  preprocessing: PreprocessingInfo;
  stage1: Stage1Info;
  stage2: Stage2Info;
  prediction: PredictionInfo;
  timing: TimingInfo;
  timeline: TimelineStep[];
  final_sentence?: string | null;
  selected_sentence_id?: string | null;
  used_fallback: boolean;
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

export interface UploadModelResult {
  saved_path: string;
  model: ModelInfo;
  profile: ProfileManifest;
}

export interface UploadEdfResult {
  saved_path: string;
  filename: string;
  group_id: string;
  role: "edf" | "marker";
  purpose: "calibration" | "inference";
  profile_id: string;
}

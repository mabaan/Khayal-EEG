# AGENTS.md

## Mission
Build Khayal as a local-only, user-facing EEG imagined speech application.

## Source of Truth
1. Khayal software design paper.
2. EEG-Preprocessing repo reference for implementation details.
3. Project constraints in this repository.

## Product Boundaries
1. Offline/local only.
2. No SQL, no cloud, no remote auth.
3. No live-stream EEG in v1; use EDF upload and simulated replay.
4. Do not abstract for arbitrary EEG protocols, arbitrary vocabularies, or arbitrary sentence lengths in v1.

## Fixed Pipeline
1. Stage 1: Diff-E only, personalized per profile.
2. Stage 2: Retrieval + Qwen reranking only, shared globally.
3. One active personalized Stage 1 checkpoint per profile.

## Readiness Gate
Inference must hard-fail unless the active profile has a completed personalized Stage 1 checkpoint.

## Fixed Sentence Assets
1. `data/sentence_catalog.json` is the canonical sentence manifest.
2. Each entry includes: sentence_id, Arabic sentence text, 3 ordered word_ids, and ordered word tokens.
3. `data/labels.json` is the canonical 25-word vocabulary map.
4. No sentence editing UI, no alternate phrase sets.

## Fixed Preprocessing Pipeline
1. Mean removal.
2. 50 Hz notch.
3. Zero-phase band-pass 0.5-32.5 Hz.
4. Channel quality validation + interpolation if needed.
5. Fixed per-word timing: rest 5 s, stimulus 5 s, imagination 6 s.
6. Sentence trial has 3 consecutive word events (48 s total):
   word1 imag [10.0,16.0], word2 imag [26.0,32.0], word3 imag [42.0,48.0].
7. Use imagination-only windows; discard first 0.5 s; retain 5.5 s:
   [10.5,16.0], [26.5,32.0], [42.5,48.0].
8. Sampling assumption 256 Hz.
9. Welch band powers: delta/theta/alpha/beta/gamma.
10. Stage 1 feature tensor: concatenate 14 EEG channels + 5 band-power channels => 19 x T.

## Fixed Stage 2 Retrieval Defaults
1. Retrieve only from fixed 12-sentence catalog.
2. Use ordered 3-slot sentence representation.
3. Build query/score from full Stage 1 posterior distributions across all 3 positions.
4. Deterministic reranking behavior.
5. If reranker returns invalid index/output, fallback to first retrieved candidate.

## Storage Rules
1. Use local filesystem under `storage/`.
2. Use JSON manifests for profile/session/training/inference metadata.
3. No ORM/database layer.

## UX Language Rules
Use user-facing labels: Session, Live Signals, Current Step, Signal Status, Predicted Sentence, Recent Sessions, Model Ready, Upload EEG Recording.

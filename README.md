# Khayal Local

Khayal Local is a local-only EEG imagined speech application for the fixed Khayal protocol. It decodes uploaded EDF sentence trials with a personalized Stage 1 DiffE checkpoint and local Stage 2 sentence selection.

## What This Repo Includes
- Next.js local UI with authenticated profile flow
- FastAPI Python inference service
- Fixed 25-word Arabic vocabulary
- Fixed 12-sentence Arabic catalog
- Stage 1 DiffE checkpoint validation and inference
- Deterministic Stage 2 retrieval over the fixed sentence catalog
- Optional local Qwen/Ollama shortlist reranking
- Local JSON manifests for profile, session, and inference records

## Product Boundaries
- Local machine only
- No SQL, no cloud storage, no remote auth
- No live EEG streaming
- Fixed Emotiv EPOC X protocol
- Fixed 3-word sentences
- Fixed 12-sentence catalog
- Inference stays blocked until the active profile has a validated personalized Stage 1 checkpoint

## Fixed Khayal Protocol
- 14 EEG channels:
  `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`
- Target sampling rate: `256 Hz`
- Preprocessing:
  - mean removal
  - 50 Hz notch
  - zero-phase band-pass `0.5-32.5 Hz`
  - channel quality validation and interpolation if needed
- Three imagination phases per trial
- For each imagination phase:
  - discard first `0.5 s`
  - keep next `5.5 s`
- Stage 1 tensor shape per word position: `[19, 1408]`
  - 14 EEG rows
  - 5 repeated Welch band-power rows
- Stage 1 returns word probabilities over the 25-word vocabulary for each of the 3 positions
- Stage 2 ranks only the fixed 12-sentence catalog

## Required Local Software
- Node.js 20+
- Python 3.10+
- `pip install -r requirements.txt`
- `npm install`
- Optional:
  - Ollama with `qwen2.5:7b-instruct`
  - local Hugging Face cache for `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Install
1. Create and activate a Python environment.
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install frontend packages:
   ```bash
   npm install
   ```

## Start The Services

### Python Service
```bash
python -m uvicorn python_service.main:app --host 127.0.0.1 --port 8001 --reload
```

### Next.js App
Windows:
```bash
set PYTHON_SERVICE_URL=http://127.0.0.1:8001
npm run dev
```

macOS / Linux:
```bash
export PYTHON_SERVICE_URL=http://127.0.0.1:8001
npm run dev
```

You can also use:
```bash
./run-local.ps1
```
or
```bash
bash ./run-local.sh
```

## App Workflow
1. Start the Python service.
2. Start the Next.js app.
3. Open the local app and log in.
4. Create or select an active profile in `Setup`.
5. Review the sentence and word assets in `Calibration`.
6. Open `Session`.
7. Upload a Stage 1 checkpoint in `.pt` or `.pth` format.
8. Upload one raw EEG `.edf` sentence trial.
9. Optionally upload the companion marker `.csv`.
10. Click `Run Khayal Inference`.
11. Review the final Arabic sentence, timeline, candidate shortlist, and local diagnostics.

The uploaded checkpoint is validated and attached to the active profile as its ready Stage 1 model.

## Expected Checkpoint Format
The Stage 1 checkpoint must be a dictionary containing:
- `model_state_dict`
- `n_classes`
- `arch`
- `channel_mean`
- `channel_std`

Expected architecture fields:
- `arch_type: diffe`
- `num_channels: 19`
- `window_size: 352`
- `n_classes: 25`

## Expected EDF And Marker Format
- EDF must contain the 14 required Emotiv EPOC X EEG channels.
- Marker CSV must contain usable `phase_Imagine` rows with `latency` and `duration`.
- A sentence trial requires exactly 3 usable imagination phases.

## Runtime Steps
1. Validate the Stage 1 checkpoint.
2. Load the EDF trial.
3. Detect or load the marker CSV.
4. Preprocess the EEG.
5. Segment the 3 imagination windows.
6. Build `[19, 1408]` Stage 1 tensors.
7. Run the DiffE classifier across non-overlapping windows.
8. Average logits and produce word probabilities for each position.
9. Build Stage 2 sentence candidates from the full three-position evidence.
10. Refine locally when optional local assets exist.
11. Rerank with local Ollama/Qwen when available.
12. Return the final decoded sentence.

## Stage 2 Fallback Behavior
Stage 2 always starts from deterministic retrieval over the 12-sentence catalog.

Optional additions:
- local retrieval refinement if assets exist
- Ollama/Qwen shortlist reranking if Ollama is available

If an optional dependency is unavailable:
- the app does not fail
- the backend returns a warning
- deterministic rank-1 retrieval is used

## Useful Repo Paths
- `app/`: Next.js pages and API routes
- `components/`: reusable UI panels
- `lib/`: local stores, API helpers, shared types
- `python_service/`: FastAPI, adapters, inference flow, service schemas
- `python_service/research_adapters/`: vendored research reference files
- `data/`: canonical vocabulary and sentence manifests
- `storage/`: local models, user data, session records, and logs

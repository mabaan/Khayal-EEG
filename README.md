# Khayal Local

Khayal Local is a local-only EEG imagined speech demo application for the fixed Khayal protocol. The current inference flow is a real thesis-defense demo path: raw EDF trial + marker CSV -> Stage 1 DiffE checkpoint -> deterministic Stage 2 sentence decoding with optional local refinement.

## What this repo includes
- Next.js local UI with authenticated profile flow
- FastAPI Python inference service
- Fixed 25-word Arabic vocabulary
- Fixed 12-sentence Arabic catalog
- Bundled S5 demo checkpoint and S5 / C7 / T2 demo trial
- Real Stage 1 DiffE inference
- Deterministic EEG-posterior Stage 2 retrieval
- Optional local transformer retrieval refinement when local assets exist
- Optional local Ollama/Qwen shortlist reranking

## Product boundaries
- Local machine only
- No SQL, no cloud storage, no remote auth
- No live EEG streaming in v1
- Fixed Emotiv EPOC X protocol
- Fixed 3-word sentences
- Fixed 12-sentence catalog
- Inference stays blocked until the active profile has a validated personalized Stage 1 checkpoint

## Fixed Khayal protocol
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
- Stage 1 tensor shape per slot: `[19, 1408]`
  - 14 EEG rows
  - 5 repeated Welch band-power rows
- Stage 1 outputs full posterior probabilities over 25 words for each of the 3 slots
- Stage 2 always starts from deterministic EEG-posterior sentence retrieval

## Required local software
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

## Start the services

### Python service
```bash
python -m uvicorn python_service.main:app --host 127.0.0.1 --port 8001 --reload
```

### Next.js app
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

## Bundled demo case
The repo now includes the exact local demo assets under:

- `storage/demo_models/S5_classifier_A.pt`
- `storage/demo_edf/S5_C7_T2/Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00.edf`
- `storage/demo_edf/S5_C7_T2/Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00_intervalMarker.csv`
- `storage/demo_edf/S5_C7_T2/Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00.json`

Expected validation target:
- Sentence ID: `C7`
- Arabic: `الطبيب يغادر المستشفى`
- Romanized: `altabeeb yughadir almustashfa`

## Demo workflow in the UI
1. Start the Python service.
2. Start the Next.js app.
3. Open the local app and log in.
4. Create or select an active profile in `Setup` if needed.
5. Open the `Session` page.
6. Click `Load Demo Case`.
7. Click `Run Khayal Inference`.
8. Review:
   - the step timeline
   - final Arabic sentence
   - Stage 1 top-k evidence for each word slot
   - Stage 2 candidate shortlist
   - debug details

## Custom upload workflow
You can also run custom local files:

1. Upload a Stage 1 checkpoint in `.pt` or `.pth` format.
2. Upload one raw EEG `.edf` sentence trial.
3. Optionally upload the companion marker `.csv`.
4. Choose Stage 2 mode:
   - `dual`
   - `deterministic`
5. Click `Run Khayal Inference`.

The uploaded checkpoint is validated and attached to the active profile as its ready Stage 1 model.

## Expected checkpoint format
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

## Expected EDF and marker format
- EDF must contain the 14 required Emotiv EPOC X EEG channels.
- Marker CSV must contain usable `phase_Imagine` rows with `latency` and `duration`.
- The live demo path requires exactly 3 usable imagination phases.

## What each pipeline step does
1. Validate the Stage 1 checkpoint.
2. Load the EDF trial.
3. Detect or load the marker CSV.
4. Preprocess the EEG.
5. Segment the 3 imagination windows.
6. Build `[19, 1408]` Stage 1 tensors.
7. Run the DiffE classifier across non-overlapping windows.
8. Average logits and produce 25-way posteriors per slot.
9. Build Stage 2 sentence candidates from the full posteriors.
10. Optionally refine locally when local assets exist.
11. Optionally rerank with local Ollama/Qwen.
12. Return the final decoded sentence and evidence panels.

## Stage 2 dual mode and fallback behavior
`dual` mode is the default. It always starts from deterministic EEG-posterior retrieval.

Optional additions:
- transformer-based local retrieval refinement if local assets exist
- Ollama/Qwen shortlist reranking if Ollama is available

If either optional dependency is unavailable:
- the app does **not** fail
- the backend returns a warning
- deterministic rank-1 retrieval is used

This means the demo still works fully offline without Ollama or Hugging Face assets.

## CLI demo test
Run the bundled demo without the frontend:

```bash
python -m python_service.test_demo_inference
```

You can also pass explicit paths:

```bash
python -m python_service.test_demo_inference \
  --model_path storage/demo_models/S5_classifier_A.pt \
  --edf_path storage/demo_edf/S5_C7_T2/Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00.edf \
  --marker_csv storage/demo_edf/S5_C7_T2/Sentence_7_EPOCX_570600_2025.11.12T04.15.10.08.00_intervalMarker.csv
```

The script prints:
- checkpoint validation
- architecture values
- marker detection
- slot tensor shapes
- Stage 1 top-k results
- Stage 2 candidate shortlist
- final prediction
- fallback status

## Useful repo paths
- `app/`: Next.js pages and API routes
- `components/`: reusable UI panels
- `lib/`: local stores, API helpers, shared types
- `python_service/`: FastAPI, adapters, inference pipeline, CLI test
- `python_service/research_adapters/`: vendored research reference files
- `data/`: canonical vocabulary and sentence manifests
- `storage/`: local models, demo files, user data, logs

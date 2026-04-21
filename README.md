# Khayal Local

Khayal Local is an offline, local-only EEG imagined speech application.

## What this v1 includes
- Next.js + TypeScript UI and local API layer
- Python FastAPI EEG pipeline service
- Fixed 12-sentence Arabic catalog and fixed 25-word vocabulary
- Fixed Diff-E Stage 1 + fixed Retrieval + Qwen Stage 2 pipeline
- First-time profile setup, calibration upload, training, session inference, and local history
- Simulated live session mode (no live EEG SDK dependency)

## Product boundaries
- Local machine only
- No SQL, no cloud storage, no remote auth
- No model chooser and no phrase-set chooser in UI
- Inference is blocked until a personalized Stage 1 checkpoint exists for the active profile

## Local requirements
- Node.js 20+
- Python 3.10+
- Pip dependencies from `requirements.txt`
- Ollama (for local Qwen reranking) if you want Stage 2 LLM reranking enabled

## Quick start (Windows)
1. Create and activate a Python virtual environment.
2. Install Python packages:
   `pip install -r requirements.txt`
3. Install Node packages:
   `npm install`
4. Put your base model at:
   `storage/base_models/diff_e_base.pt`
5. Start both services:
   `./run-local.ps1`

## Quick start (macOS/Linux)
1. Create and activate a Python virtual environment.
2. Install Python packages:
   `pip install -r requirements.txt`
3. Install Node packages:
   `npm install`
4. Put your base model at:
   `storage/base_models/diff_e_base.pt`
5. Start both services:
   `bash ./run-local.sh`

## Manual service start
- Python service:
  `python -m uvicorn python_service.main:app --host 127.0.0.1 --port 8001 --reload`
- Next app:
  `set PYTHON_SERVICE_URL=http://127.0.0.1:8001` (Windows)
  `export PYTHON_SERVICE_URL=http://127.0.0.1:8001` (Unix)
  `npm run dev`

## Key folders
- `app/`: Next.js app router pages and local API routes
- `components/`: reusable UI components
- `lib/`: app constants, stores, validation, and helper logic
- `python_service/`: preprocessing, segmentation, training, inference, retrieval, rerank
- `data/`: fixed sentence/vocabulary/prompt/default manifests
- `storage/`: local user/session/model/log data

## Notes
- Stage 2 reranking defaults to Ollama `http://127.0.0.1:11434`.
- If Ollama is unavailable, the pipeline falls back to deterministic retrieval candidate #1.
- This repo is intentionally hardcoded for one protocol, one vocabulary, and one sentence shape.

## Frontend-only run (no Python, no Ollama)
- `npm run dev`
- or `npm run dev:frontend`
- or `./run-local.ps1 -FrontendOnly`

This starts only Next.js on `http://127.0.0.1:3000`.

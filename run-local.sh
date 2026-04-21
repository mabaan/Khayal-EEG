#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_PORT=8001
APP_PORT=3000

cd "$ROOT_DIR"

PY_PID=""

if [[ "${FRONTEND_ONLY:-0}" != "1" ]]; then
  if command -v python >/dev/null 2>&1; then
    python -m uvicorn python_service.main:app --host 127.0.0.1 --port "$PY_PORT" --reload &
    PY_PID=$!
    export NEXT_PUBLIC_PYTHON_SERVICE_URL="http://127.0.0.1:${PY_PORT}"
    export PYTHON_SERVICE_URL="http://127.0.0.1:${PY_PORT}"
  else
    echo "[warn] python not found, continuing in frontend-only mode"
  fi
fi

cleanup() {
  if [[ -n "$PY_PID" ]]; then
    kill "$PY_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

next dev -p "$APP_PORT"

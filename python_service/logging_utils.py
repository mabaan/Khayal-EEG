from __future__ import annotations

from datetime import datetime

from .config import LOGS_DIR


def _stamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def append_log(file_name: str, line: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / file_name
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_stamp()}] {line}\n")

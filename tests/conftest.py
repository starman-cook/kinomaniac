"""Общие фикстуры для pytest: изоляция calls.jsonl от тестовых запусков."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_calls_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет запись calls.jsonl во временный файл, чтобы тесты не засоряли журнал."""
    log_file = tmp_path / "calls.jsonl"
    monkeypatch.setenv("KINO_CALLS_LOG", str(log_file))
    try:
        import kino

        if hasattr(kino, "CALLS_LOG_PATH"):
            monkeypatch.setattr(kino, "CALLS_LOG_PATH", str(log_file))
    except ImportError:
        pass
    return log_file

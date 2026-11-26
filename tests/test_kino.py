"""Тесты уровня Common (REQ-C01..REQ-C10) с заглушкой клиента без вызова API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from pydantic import ValidationError

from kino import PickItem, Selection, build_prompt, pick


def make_response(
    parsed=None,
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
):
    message = SimpleNamespace(parsed=parsed, refusal=refusal, content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def make_client(response=None, *, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.parse.side_effect = side_effect
    else:
        client.chat.completions.parse.return_value = response
    return client


FILMS = [
    {
        "id": "f1",
        "title": "Каталожный фильм А",
        "year": 2000,
        "minutes": 100,
        "genres": ["драма"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "Описание строго из каталога А.",
    },
    {
        "id": "f2",
        "title": "Каталожный фильм Б",
        "year": 2005,
        "minutes": 90,
        "genres": ["комедия"],
        "countries": ["Казахстан"],
        "languages": ["казахский"],
        "description": "Описание строго из каталога Б.",
    },
    {
        "id": "f3",
        "title": "Каталожный фильм В",
        "year": 2010,
        "minutes": 115,
        "genres": ["боевик"],
        "countries": ["Россия"],
        "languages": ["русский"],
        "description": "Описание строго из каталога В.",
    },
    {
        "id": "f4",
        "title": "Каталожный фильм Г",
        "year": 2018,
        "minutes": 130,
        "genres": ["фантастика"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "Описание строго из каталога Г.",
    },
]


def test_prompt_contains_tags_rule_and_example():
    prompt = build_prompt("тестовый запрос", FILMS)
    assert "<catalog>" in prompt and "</catalog>" in prompt
    assert "<query>" in prompt and "</query>" in prompt
    assert "тестовый запрос" in prompt
    assert "данные" in prompt.lower()
    assert "film_id" in prompt and "status" in prompt


def test_valid_pick_prints_catalog_data_and_logs(
    capsys: pytest.CaptureFixture[str], isolate_calls_log: Path
):
    parsed = Selection(
        status="ok",
        reason="Подходит по настроению",
        picks=[PickItem(film_id="f1", why="Сильная драма")],
    )
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "gpt-6-luna", "драма", FILMS)

    assert outcome == "ok"
    assert ids == ["f1"]

    out = capsys.readouterr().out
    assert "Каталожный фильм А" in out
    assert "2000" in out
    assert "100" in out
    assert "Описание строго из каталога А." in out
    assert "Сильная драма" in out

    lines = isolate_calls_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == "gpt-6-luna"
    assert entry["outcome"] == "ok"
    assert entry["request"]["query"] == "драма"
    assert entry["request"]["catalog_ids"] == ["f1", "f2", "f3", "f4"]
    assert entry["usage"]["total_tokens"] == 160


def test_alien_id_rejected_and_not_shown(
    capsys: pytest.CaptureFixture[str], isolate_calls_log: Path
):
    parsed = Selection(
        status="ok",
        reason="Нашёл фильм",
        picks=[PickItem(film_id="f999", why="Выдуманный фильм")],
    )
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "test-model", "query", FILMS)

    assert outcome == "invalid"
    assert ids == []
    out = capsys.readouterr().out
    assert "f999" not in out
    assert "Выдуманный фильм" not in out

    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["outcome"] == "invalid"
    assert entry["rejected"] == ["f999"]


def test_duplicate_id_rejected(isolate_calls_log: Path):
    parsed = Selection(
        status="ok",
        reason="Повтор",
        picks=[
            PickItem(film_id="f1", why="первый раз"),
            PickItem(film_id="f1", why="второй раз"),
        ],
    )
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "test-model", "query", FILMS)

    assert outcome == "invalid"
    assert ids == []
    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["outcome"] == "invalid"
    assert entry["rejected"] == ["f1"]


def test_more_than_three_valid_picks_capped_at_three():
    parsed = Selection(
        status="ok",
        reason="Четыре фильма",
        picks=[
            PickItem(film_id="f1", why="1"),
            PickItem(film_id="f2", why="2"),
            PickItem(film_id="f3", why="3"),
            PickItem(film_id="f4", why="4"),
        ],
    )
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "test-model", "query", FILMS)

    assert outcome == "ok"
    assert ids == ["f1", "f2", "f3"]


def test_fourth_pick_alien_is_still_caught(isolate_calls_log: Path):
    parsed = Selection(
        status="ok",
        reason="Четвёртый чужой",
        picks=[
            PickItem(film_id="f1", why="1"),
            PickItem(film_id="f2", why="2"),
            PickItem(film_id="f3", why="3"),
            PickItem(film_id="f999", why="чужой"),
        ],
    )
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "test-model", "query", FILMS)

    assert outcome == "invalid"
    assert ids == []
    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["rejected"] == ["f999"]


def test_ok_with_empty_picks_is_invalid(isolate_calls_log: Path):
    parsed = Selection(status="ok", reason="пусто", picks=[])
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "test-model", "query", FILMS)

    assert outcome == "invalid"
    assert ids == []
    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["outcome"] == "invalid"
    assert "rejected" in entry


def test_no_match():
    parsed = Selection(status="no_match", reason="В каталоге нет подходящих фильмов", picks=[])
    client = make_client(make_response(parsed))

    outcome, ids = pick(client, "test-model", "мультфильм 2031 года", FILMS)

    assert outcome == "no_match"
    assert ids == []


def test_refusal_from_message_field():
    client = make_client(make_response(parsed=None, refusal="I cannot fulfill this request"))

    outcome, ids = pick(client, "test-model", "запрещённый запрос", FILMS)

    assert outcome == "refusal"
    assert ids == []


def test_length_from_finish_reason():
    client = make_client(make_response(parsed=None, finish_reason="length"))

    outcome, ids = pick(client, "test-model", "запрос", FILMS)

    assert outcome == "length"
    assert ids == []


def test_no_connection_error():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    client = make_client(side_effect=openai.APIConnectionError(request=req))

    outcome, ids = pick(client, "test-model", "запрос", FILMS)

    assert outcome == "no_connection"
    assert ids == []


def test_api_status_error_includes_http_code():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    exc = openai.RateLimitError("rate limited", response=resp, body=None)
    client = make_client(side_effect=exc)

    outcome, ids = pick(client, "test-model", "запрос", FILMS)

    assert outcome == "error_429"
    assert ids == []


def test_validation_error_returns_invalid(isolate_calls_log: Path):
    try:
        Selection.model_validate({"status": "broken"})
    except ValidationError as exc:
        val_err = exc

    client = make_client(side_effect=val_err)

    outcome, ids = pick(client, "test-model", "запрос", FILMS)

    assert outcome == "invalid"
    assert ids == []
    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["outcome"] == "invalid"
    assert "rejected" in entry


def test_qwen_passes_reasoning_effort_none():
    parsed = Selection(status="no_match", reason="нет", picks=[])
    client = make_client(make_response(parsed))

    pick(client, "qwen3.5:4b", "запрос", FILMS)

    _, kwargs = client.chat.completions.parse.call_args
    assert kwargs.get("extra_body") == {"reasoning_effort": "none"}

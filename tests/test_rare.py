"""Тесты уровня Rare (REQ-R01..REQ-R05): фильтры и проверка условий кодом."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from kino import Filters, PickItem, Selection, apply_filters, pick_filtered


def make_response(parsed):
    message = SimpleNamespace(parsed=parsed, refusal=None, content=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=80, completion_tokens=20, total_tokens=100)
    return SimpleNamespace(choices=[choice], usage=usage)


CATALOG = [
    {
        "id": "kz-04",
        "title": "Бизнес по-казахски",
        "year": 2016,
        "minutes": 90,
        "genres": ["комедия"],
        "countries": ["Казахстан"],
        "languages": ["казахский", "русский"],
        "description": "Комедия о гостинице.",
    },
    {
        "id": "kz-05",
        "title": "Томирис",
        "year": 2019,
        "minutes": 156,
        "genres": ["исторический", "драма", "боевик"],
        "countries": ["Казахстан"],
        "languages": ["казахский", "русский"],
        "description": "Историческая драма.",
    },
    {
        "id": "f-029",
        "title": "Ла-Ла Ленд",
        "year": 2016,
        "minutes": 128,
        "genres": ["мюзикл", "драма", "мелодрама", "комедия"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "Романтический мюзикл.",
    },
    {
        "id": "f-004",
        "title": "Брат",
        "year": 1997,
        "minutes": 96,
        "genres": ["криминал", "драма", "боевик"],
        "countries": ["Россия"],
        "languages": ["русский"],
        "description": "Криминальная драма.",
    },
]


def test_apply_filters_by_language_and_max_minutes():
    flt = Filters(languages=["казахский"], max_minutes=120)
    result = apply_filters(CATALOG, flt)
    assert [f["id"] for f in result] == ["kz-04"]


def test_apply_filters_by_years_and_exclude_genres():
    flt = Filters(year_from=2010, year_to=2020, must_genres=["комедия"], exclude_genres=["мелодрама"])
    result = apply_filters(CATALOG, flt)
    assert [f["id"] for f in result] == ["kz-04"]


def test_pick_filtered_two_steps_and_logs_steps(isolate_calls_log: Path):
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        make_response(Filters(languages=["казахский"], max_minutes=120)),
        make_response(
            Selection(
                status="ok",
                reason="Короткая казахская комедия",
                picks=[PickItem(film_id="kz-04", why="90 минут и есть казахский язык")],
            )
        ),
    ]

    outcome, ids = pick_filtered(client, "gpt-6-luna", "на казахском до двух часов", CATALOG)

    assert outcome == "ok"
    assert ids == ["kz-04"]
    assert client.chat.completions.parse.call_count == 2

    lines = [
        json.loads(line)
        for line in isolate_calls_log.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [entry["step"] for entry in lines] == ["filters", "choice"]


def test_pick_filtered_returns_no_match_when_code_filter_empties_catalog(isolate_calls_log: Path):
    client = MagicMock()
    client.chat.completions.parse.return_value = make_response(Filters(year_from=2030))

    outcome, ids = pick_filtered(client, "gpt-6-luna", "фильм 2031 года", CATALOG)

    assert outcome == "no_match"
    assert ids == []
    assert client.chat.completions.parse.call_count == 1
    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["step"] == "filters"


def test_pick_filtered_post_check_rejects_filter_violation(isolate_calls_log: Path):
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        make_response(Filters(languages=["казахский"], max_minutes=120)),
        make_response(
            Selection(
                status="ok",
                reason="Нарушил лимит",
                picks=[PickItem(film_id="kz-05", why="156 минут, больше 120")],
            )
        ),
    ]

    outcome, ids = pick_filtered(client, "gpt-6-luna", "на казахском до двух часов", CATALOG)

    assert outcome == "invalid"
    assert ids == []
    lines = [
        json.loads(line)
        for line in isolate_calls_log.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert lines[-1]["step"] == "choice"
    assert lines[-1]["outcome"] == "invalid"
    assert lines[-1]["rejected"] == ["kz-05"]

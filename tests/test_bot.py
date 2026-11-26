"""Тесты уровня Legendary (REQ-L01..REQ-L09): Telegram-бот с диалогом.

Тесты написаны ДО реализации bot.py по процессу лекции 5.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from bot import BotPickResponse, MovieDialogEngine, extract_exclusions, is_more_request
from kino import PickItem


def make_response(parsed, *, refusal: str | None = None, finish_reason: str = "stop"):
    message = SimpleNamespace(parsed=parsed, refusal=refusal, content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130)
    return SimpleNamespace(choices=[choice], usage=usage)


FILMS = [
    {
        "id": "f-004",
        "title": "Брат",
        "year": 1997,
        "minutes": 96,
        "genres": ["криминал", "драма", "боевик"],
        "countries": ["Россия"],
        "languages": ["русский"],
        "description": "Данила Багров приезжает в Петербург.",
    },
    {
        "id": "kz-01",
        "title": "Рэкетир",
        "year": 2007,
        "minutes": 80,
        "genres": ["криминал", "драма", "боевик"],
        "countries": ["Казахстан"],
        "languages": ["русский"],
        "description": "Боксёр Саян в Алма-Ате 90-х.",
    },
    {
        "id": "f-010",
        "title": "Титаник",
        "year": 1997,
        "minutes": 194,
        "genres": ["драма", "мелодрама"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "История любви на борту Титаника.",
    },
    {
        "id": "kz-04",
        "title": "Бизнес по-казахски",
        "year": 2016,
        "minutes": 90,
        "genres": ["комедия"],
        "countries": ["Казахстан"],
        "languages": ["казахский", "русский"],
        "description": "Комедия про гостиницу и родственников.",
    },
]


def test_is_more_request_variants():
    assert is_more_request("Ещё")
    assert is_more_request("ещё")
    assert is_more_request("еще")
    assert is_more_request("more")
    assert is_more_request("MORE!")
    assert not is_more_request("посоветуй комедию")


def test_extract_exclusions_matches_catalog_titles():
    assert extract_exclusions("уже смотрел Брат", FILMS) == ["f-004"]
    assert extract_exclusions("не показывай Титаник и Рэкетир", FILMS) == ["kz-01", "f-010"]
    assert extract_exclusions("посоветуй комедию", FILMS) == []


def test_vague_query_returns_clarify_with_single_question():
    client = MagicMock()
    client.chat.completions.parse.return_value = make_response(
        BotPickResponse(
            status="clarify",
            reason="Запрос слишком общий",
            question="Какое настроение или жанр вам сейчас ближе?",
            picks=[],
        )
    )
    engine = MovieDialogEngine(client, "gpt-6-luna", FILMS)

    result = engine.handle_message(chat_id=1, text="посоветуй что-нибудь")

    assert result.status == "clarify"
    assert result.ids == []
    assert "Какое настроение" in result.text
    assert engine.get_session(1).pending_query == "посоветуй что-нибудь"


def test_clarify_followup_combines_query_and_picks_from_catalog():
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        make_response(
            BotPickResponse(
                status="clarify",
                reason="Слишком размыто",
                question="Какой жанр предпочитаете?",
                picks=[],
            )
        ),
        make_response(
            BotPickResponse(
                status="ok",
                reason="Подборка комедий",
                picks=[PickItem(film_id="kz-04", why="Лёгкая казахская комедия")],
            )
        ),
    ]
    engine = MovieDialogEngine(client, "gpt-6-luna", FILMS)

    first = engine.handle_message(chat_id=10, text="хочу фильм на вечер")
    assert first.status == "clarify"

    second = engine.handle_message(chat_id=10, text="комедию")
    assert second.status == "ok"
    assert second.ids == ["kz-04"]
    assert "Бизнес по-казахски" in second.text
    assert "2016" in second.text
    assert "90" in second.text
    assert engine.get_session(10).pending_query is None


def test_more_returns_new_films_without_repeats_until_exhausted():
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        make_response(
            BotPickResponse(
                status="ok",
                reason="Первая подборка",
                picks=[PickItem(film_id="f-004", why="Культовый боевик")],
            )
        ),
        make_response(
            BotPickResponse(
                status="ok",
                reason="Новая подборка",
                picks=[PickItem(film_id="kz-01", why="Похож по атмосфере 90-х")],
            )
        ),
    ]
    engine = MovieDialogEngine(client, "gpt-6-luna", FILMS[:2])

    res1 = engine.handle_message(chat_id=2, text="криминальный боевик")
    assert res1.status == "ok"
    assert res1.ids == ["f-004"]

    res2 = engine.handle_message(chat_id=2, text="Ещё")
    assert res2.status == "ok"
    assert res2.ids == ["kz-01"]
    assert "f-004" not in res2.ids

    # Проверяем, что во второй вызов модели уже показанный f-004 не попал в каталог
    second_call_prompt = client.chat.completions.parse.call_args_list[1].kwargs["messages"][0]["content"]
    assert '"id": "f-004"' not in second_call_prompt

    # Все фильмы из FILMS[:2] показаны — третье «more» должно честно вернуть no_match без повторов
    res3 = engine.handle_message(chat_id=2, text="more")
    assert res3.status == "no_match"
    assert res3.ids == []


def test_exclusions_persist_across_dialog_turns_and_reset_clears_them():
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        make_response(
            BotPickResponse(
                status="ok",
                reason="Подобрал без Брата",
                picks=[PickItem(film_id="kz-01", why="Подходит под криминал")],
            )
        ),
        make_response(
            BotPickResponse(
                status="ok",
                reason="После сброса",
                picks=[PickItem(film_id="f-004", why="Теперь снова доступен")],
            )
        ),
    ]
    engine = MovieDialogEngine(client, "gpt-6-luna", FILMS)

    # Пользователь исключает «Брат»
    ack = engine.handle_message(chat_id=3, text="уже смотрел Брат")
    assert "f-004" in engine.get_session(3).excluded_ids
    assert "Брат" in ack.text

    # На следующем запросе «Брат» исключён из каталога кандидатов
    res = engine.handle_message(chat_id=3, text="криминальная драма")
    assert res.status == "ok"
    assert res.ids == ["kz-01"]
    prompt_sent = client.chat.completions.parse.call_args_list[0].kwargs["messages"][0]["content"]
    assert '"id": "f-004"' not in prompt_sent

    # После сброса диалога исключение снято
    engine.reset_session(3)
    assert engine.get_session(3).excluded_ids == set()
    res_after_reset = engine.handle_message(chat_id=3, text="криминальная драма")
    assert res_after_reset.status == "ok"
    assert res_after_reset.ids == ["f-004"]


def test_bot_rejects_alien_or_excluded_id_from_model(isolate_calls_log: Path):
    client = MagicMock()
    client.chat.completions.parse.return_value = make_response(
        BotPickResponse(
            status="ok",
            reason="Попытка вернуть чужой id",
            picks=[PickItem(film_id="fake-777", why="Выдумка")],
        )
    )
    engine = MovieDialogEngine(client, "gpt-6-luna", FILMS)

    res = engine.handle_message(chat_id=4, text="что-нибудь крутое")

    assert res.status == "invalid"
    assert res.ids == []
    assert "fake-777" not in res.text
    entry = json.loads(isolate_calls_log.read_text(encoding="utf-8").strip())
    assert entry["outcome"] == "invalid"
    assert entry["rejected"] == ["fake-777"]


def test_bot_qwen_local_passes_reasoning_effort_none():
    client = MagicMock()
    client.chat.completions.parse.return_value = make_response(
        BotPickResponse(status="no_match", reason="нет", picks=[])
    )
    engine = MovieDialogEngine(client, "qwen3.5:4b", FILMS)

    engine.handle_message(chat_id=5, text="французский нуар")

    _, kwargs = client.chat.completions.parse.call_args
    assert kwargs.get("extra_body") == {"reasoning_effort": "none"}

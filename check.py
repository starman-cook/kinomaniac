"""Прогон набора queries.json (Common / Rare / Legendary stability) и опыт с памятью модели.

Примеры запуска:
    uv run check.py
    uv run check.py --mode rare --runs 3
    uv run check.py --mode rare --runs 5
    uv run check.py --memory-test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import kino
from kino import load_films, pick, pick_filtered


class FilmMemoryAnswer(BaseModel):
    year: int
    minutes: int


def load_queries(path: str = "queries.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_expectation(outcome: str, ids: list[str], expect: dict | None) -> bool:
    """Проверяет, совпал ли результат с ожиданием (SPEC 4.6 и 6.2)."""
    if not expect:
        return True

    if expect.get("no_match"):
        return outcome == "no_match" and ids == []

    if outcome != "ok":
        return False

    must = set(expect.get("must", []))
    forbid = set(expect.get("forbid", []))

    if must and not (must & set(ids)):
        return False
    if forbid and (forbid & set(ids)):
        return False
    return True


def _count_log_lines(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _read_new_entries(path: str, since_line: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    entries: list[dict] = []
    with p.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[since_line:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def run_memory_experiment(client: OpenAI, model: str, films: list[dict], count: int = 10) -> None:
    """Опыт с памятью модели (SPEC 5.9): год и длительность 10 фильмов БЕЗ каталога."""
    sample = films[:count]
    extra = {"extra_body": {"reasoning_effort": "none"}} if "qwen" in model.lower() else {}

    print(f"Опыт с памятью модели ({model}) — {len(sample)} фильмов БЕЗ каталога:\n")
    year_mismatches = 0
    minutes_mismatches = 0
    any_mismatches = 0

    for film in sample:
        prompt = (
            f"Назови точный год выхода и длительность в минутах фильма «{film['title']}» "
            f"(страна: {', '.join(film.get('countries', []))})."
        )
        try:
            resp = client.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=FilmMemoryAnswer,
                **extra,
            )
            ans = resp.choices[0].message.parsed
            usage = getattr(resp, "usage", None)
            kino._log(
                model,
                {"query": prompt, "film_id": film["id"]},
                "ok" if ans else "invalid",
                usage=usage,
                step="memory",
            )
        except Exception as exc:
            outcome = kino._classify_error(exc)
            kino._log(model, {"query": prompt, "film_id": film["id"]}, outcome, step="memory")
            print(f"  {film['title']}: ошибка {outcome}")
            any_mismatches += 1
            continue

        if ans is None:
            print(f"  {film['title']}: не разобран ответ")
            any_mismatches += 1
            continue

        y_ok = ans.year == film["year"]
        m_ok = ans.minutes == film["minutes"]
        if not y_ok:
            year_mismatches += 1
        if not m_ok:
            minutes_mismatches += 1
        if not (y_ok and m_ok):
            any_mismatches += 1

        mark = "OK" if (y_ok and m_ok) else "DIFF"
        print(
            f"  [{mark}] {film['title']}: модель=({ans.year}, {ans.minutes} мин) "
            f"vs каталог=({film['year']}, {film['minutes']} мин)"
        )

    print("\n" + "=" * 50)
    print(f"Расхождений по году:         {year_mismatches}/{len(sample)}")
    print(f"Расхождений по длительности: {minutes_mismatches}/{len(sample)}")
    print(f"Фильмов с любым расхождением:{any_mismatches}/{len(sample)}")


def run_eval(
    client: OpenAI,
    model: str,
    films: list[dict],
    queries: list[dict],
    *,
    mode: str = "common",
    runs: int = 1,
    quiet: bool = False,
) -> dict:
    picker = pick_filtered if mode == "rare" else pick
    per_query_pass_counts = [0] * len(queries)
    outcomes = Counter()
    alien_caught = 0
    expected_passed_total = 0
    expected_checks_total = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    t0 = time.perf_counter()

    for run_idx in range(1, runs + 1):
        run_passed = 0
        run_expected = 0
        print(f"=== Прогон {run_idx}/{runs} | модель={model} | режим={mode} | запросов={len(queries)} ===\n")
        for idx, item in enumerate(queries):
            query = item["query"]
            expect = item.get("expect")
            if expect:
                run_expected += 1
                expected_checks_total += 1

            before = _count_log_lines(kino.CALLS_LOG_PATH)
            if not quiet:
                print(f"[{idx + 1}/{len(queries)}] {query}")
            outcome, ids = picker(client, model, query, films)
            new_entries = _read_new_entries(kino.CALLS_LOG_PATH, before)

            outcomes[outcome] += 1
            for entry in new_entries:
                alien_caught += len(entry.get("rejected") or [])
                usage = entry.get("usage") or {}
                total_prompt_tokens += usage.get("prompt_tokens") or 0
                total_completion_tokens += usage.get("completion_tokens") or 0

            passed = check_expectation(outcome, ids, expect)
            if expect and passed:
                run_passed += 1
                expected_passed_total += 1
                per_query_pass_counts[idx] += 1
            elif not expect and passed:
                per_query_pass_counts[idx] += 1

            status = "OK" if passed else "FAIL"
            print(f"  -> {outcome} {ids}  [{status}]\n")

        pct = (run_passed / run_expected * 100) if run_expected else 0.0
        print(f"Итог прогона {run_idx}: пройдено {run_passed}/{run_expected} ({pct:.1f}%)\n")

    elapsed = time.perf_counter() - t0
    all_runs_passed = sum(1 for c in per_query_pass_counts if c == runs)

    print("=" * 60)
    print(f"Сводка ({model}, режим={mode}, прогонов={runs}):")
    print(f"  ok:                         {outcomes['ok']}")
    print(f"  no_match:                   {outcomes['no_match']}")
    print(f"  invalid:                    {outcomes['invalid']}")
    other = sum(v for k, v in outcomes.items() if k not in ("ok", "no_match", "invalid"))
    print(f"  другие исходы:              {other}")
    print(f"  чужих/отклонённых id:       {alien_caught}")
    rate = (expected_passed_total / expected_checks_total * 100) if expected_checks_total else 0.0
    print(f"  запросов с ожиданием:       {expected_passed_total}/{expected_checks_total} ({rate:.1f}%)")
    if runs > 1:
        stab_rate = all_runs_passed / len(queries) * 100 if queries else 0.0
        print(f"  прошли во всех {runs} прогонах: {all_runs_passed}/{len(queries)} ({stab_rate:.1f}%)")
    print(f"  токены (prompt/completion): {total_prompt_tokens} / {total_completion_tokens}")
    print(f"  время:                      {elapsed:.1f} с")

    return {
        "model": model,
        "mode": mode,
        "runs": runs,
        "outcomes": dict(outcomes),
        "alien_caught": alien_caught,
        "expected_passed": expected_passed_total,
        "expected_total": expected_checks_total,
        "all_runs_passed": all_runs_passed,
        "total_queries": len(queries),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Проверка Киноманьяка на queries.json")
    parser.add_argument("--mode", choices=["common", "rare"], default="common")
    parser.add_argument("--model", default=None, help="Переопределить MODEL из .env")
    parser.add_argument("--queries", default="queries.json", help="Путь к файлу запросов")
    parser.add_argument("--runs", type=int, default=1, help="Число прогонов")
    parser.add_argument("--limit", type=int, default=None, help="Взять первые N запросов")
    parser.add_argument("--memory-test", action="store_true", help="Запустить опыт с памятью модели")
    parser.add_argument("--memory-count", type=int, default=10, help="Сколько фильмов проверить в опыте с памятью")
    args = parser.parse_args()

    films = load_films()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    model = args.model or os.getenv("MODEL", "qwen3.5:4b")

    client = OpenAI(base_url=base_url, api_key=api_key)

    if args.memory_test:
        run_memory_experiment(client, model, films, count=args.memory_count)
        return

    queries = load_queries(args.queries)
    if args.limit is not None:
        queries = queries[: args.limit]

    run_eval(client, model, films, queries, mode=args.mode, runs=args.runs)


if __name__ == "__main__":
    main()

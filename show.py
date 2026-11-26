"""Статистика по журналу calls.jsonl: исходы, шаги, отклонённые id, токены и цена.

Запуск:
    uv run show.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

# Тарифы за 1M токенов (input, output) в USD.
# Локальные модели (qwen / ollama) стоят $0.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.4-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}
DEFAULT_CLOUD_PRICE = (0.10, 0.50)


def price_for_model(model: str) -> tuple[float, float]:
    lower = model.lower()
    if "qwen" in lower or "ollama" in lower or "llama" in lower:
        return (0.0, 0.0)
    for prefix, rates in MODEL_PRICING.items():
        if lower.startswith(prefix):
            return rates
    return DEFAULT_CLOUD_PRICE


def load_calls(path: str = "calls.jsonl") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    calls: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return calls


def main() -> None:
    calls = load_calls()
    if not calls:
        print("Журнал calls.jsonl пуст или не найден.")
        return

    outcomes = Counter(c.get("outcome", "unknown") for c in calls)
    steps = Counter(c.get("step", "common") for c in calls)
    models = Counter(c.get("model", "?") for c in calls)
    rejected_total = sum(len(c.get("rejected") or []) for c in calls)

    per_model_tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_cost = 0.0

    for c in calls:
        model = c.get("model", "?")
        usage = c.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        tt = usage.get("total_tokens") or (pt + ct)
        total_prompt += pt
        total_completion += ct
        total_tokens += tt
        per_model_tokens[model][0] += pt
        per_model_tokens[model][1] += ct
        per_model_tokens[model][2] += tt

        in_rate, out_rate = price_for_model(model)
        total_cost += pt / 1_000_000 * in_rate + ct / 1_000_000 * out_rate

    print(f"Вызовов всего: {len(calls)}")
    print(f"Модели:        {dict(models)}")
    print(f"Шаги:          {dict(steps)}")
    print(f"Отклонённых id:{rejected_total}")
    print()
    print("Исходы:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:15} {count}")
    print()
    print(f"Токены prompt:     {total_prompt}")
    print(f"Токены completion: {total_completion}")
    print(f"Токены всего:      {total_tokens}")
    print()
    print("По моделям:")
    for model, (pt, ct, tt) in per_model_tokens.items():
        in_rate, out_rate = price_for_model(model)
        cost = pt / 1_000_000 * in_rate + ct / 1_000_000 * out_rate
        avg_cost = cost / models[model] if models[model] else 0.0
        print(
            f"  {model:18} вызовов={models[model]:3d} | tokens={tt:7d} "
            f"| всего=${cost:.5f} | ср./вызов=${avg_cost:.6f}"
        )
    print()
    print(f"Примерная общая стоимость: ${total_cost:.5f} (локальная qwen3.5:4b = $0)")


if __name__ == "__main__":
    main()

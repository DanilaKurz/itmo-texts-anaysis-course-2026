"""Разовое сравнение: биграммная и триграммная модели, Add-k и Kneser-Ney.

Вопрос для защиты лаби 1 — «а что было бы при N=3?». Скрипт считает оба
порядка на тех же токенах (``artifacts/lab1/tokens_{train,test}.txt``),
теми же сглаживаниями и тем же критерием словаря (min_count=2), что и
``BigramLM``, и печатает таблицу перплексий.

Запуск из каталога ``danila-kurz``::

    python sandbox/lab1_trigram.py

Результат: таблица в консоль и ``lab1/artifacts/trigram_metrics.json``.
Память: ~3-4 ГБ на триграммные счётчики, время полного прогона ~5-10 минут.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.textkit.lm import END, START, UNK, iter_tokens

ART = ROOT / "lab1" / "artifacts"
OUT = ART / "trigram_metrics.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ``lab1/run.py`` в этом репозитории пишет тяжёлые файлы в ``artifacts/lab1``;
# если токены уже там, читаем их, иначе ищем рядом с отчётом.
TOKEN_DIRS = (ROOT / "artifacts" / "lab1", ART)


def find_tokens(name: str) -> Path:
    for directory in TOKEN_DIRS:
        path = directory / name
        if path.exists():
            return path
    raise SystemExit(f"нет {name}: сначала запустите lab1/run.py")


TRAIN = find_tokens("tokens_train.txt")
TEST = find_tokens("tokens_test.txt")
MIN_COUNT = 2
ADD_K = 0.01


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f} c] {msg}", flush=True)


def estimate_discount(ngram_counts) -> tuple[float, int, int]:
    """D = n1 / (n1 + 2*n2) по числу типов с частотой 1 и 2 (как в lm.py)."""
    n1 = n2 = 0
    for inner in ngram_counts.values():
        for v in inner.values():
            if v == 1:
                n1 += 1
            elif v == 2:
                n2 += 1
    return (n1 / (n1 + 2 * n2) if n1 and n2 else 0.5), n1, n2


def main() -> None:
    t0 = time.time()

    # Проход 1: словарь тем же критерием, что в BigramLM.fit.
    counts: Counter[str] = Counter()
    for tokens in iter_tokens(TRAIN):
        counts.update(tokens)
    vocab = {w for w, c in counts.items() if c >= MIN_COUNT}
    vocab |= {START, END, UNK}
    vocab_size = len(vocab)
    log(f"словарь: {vocab_size} типов", t0)

    # Проход 2: биграммы и триграммы одним махом.
    bigram: dict[str, dict[str, int]] = defaultdict(dict)
    trigram: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for tokens in iter_tokens(TRAIN):
        mapped = [sys.intern(w) if w in vocab else UNK for w in tokens]
        prev = mapped[0]
        for i in range(1, len(mapped)):
            cur = mapped[i]
            row = bigram[prev]
            row[cur] = row.get(cur, 0) + 1
            if i >= 2:
                ctx = trigram[(mapped[i - 2], prev)]
                ctx[cur] = ctx.get(cur, 0) + 1
            prev = cur
    bigram = dict(bigram)
    trigram = dict(trigram)
    bigram_types = sum(len(c) for c in bigram.values())
    trigram_types = sum(len(c) for c in trigram.values())
    log(f"типов биграмм: {bigram_types}, типов триграмм: {trigram_types}", t0)

    # Структуры биграммного уровня — как в lm.py, чтобы биграммные числа
    # совпали с metrics.json один в один.
    left_total = {w: sum(c.values()) for w, c in bigram.items()}
    left_types = {w: len(c) for w, c in bigram.items()}
    continuations: Counter[str] = Counter()
    for cont in bigram.values():
        for nxt in cont:
            continuations[nxt] += 1
    cont_total = sum(continuations.values())
    p_cont = {w: c / cont_total for w, c in continuations.items()}

    # Суммарные счётчики контекстов триграмм (число продолжений — len(ctx)).
    ctx_total = {ctx: sum(c.values()) for ctx, c in trigram.items()}

    d2, b1, b2 = estimate_discount(bigram)
    d3, t1, t2 = estimate_discount(trigram)
    log(f"дисконты: D2={d2:.4f} (n1={b1}, n2={b2}), D3={d3:.4f} (n1={t1}, n2={t2})", t0)

    def prob2_kn(w1: str, w2: str) -> float:
        cont = bigram.get(w1)
        if not cont or w1 not in left_total:
            return p_cont.get(w2, 1e-12)
        observed = cont.get(w2, 0)
        base = max(observed - d2, 0.0) / left_total[w1]
        backoff = d2 * left_types[w1] / left_total[w1]
        return base + backoff * p_cont.get(w2, 1e-12)

    def prob2_addk(w1: str, w2: str) -> float:
        cont = bigram.get(w1)
        num = (cont.get(w2, 0) if cont else 0) + ADD_K
        den = left_total.get(w1, 0) + ADD_K * vocab_size
        return num / den

    def prob3_kn(w1: str, w2: str, w3: str) -> float:
        # Интерполированный KN: P3 = max(c3-D3,0)/c(ctx) + lam(ctx)*P2_KN.
        ctx = trigram.get((w1, w2))
        if not ctx:
            return prob2_kn(w2, w3)
        observed = ctx.get(w3, 0)
        total = ctx_total[(w1, w2)]
        base = max(observed - d3, 0.0) / total
        lam = d3 * len(ctx) / total
        return base + lam * prob2_kn(w2, w3)

    def prob3_addk(w1: str, w2: str, w3: str) -> float:
        ctx = trigram.get((w1, w2))
        num = (ctx.get(w3, 0) if ctx else 0) + ADD_K
        den = (ctx_total[(w1, w2)] if ctx else 0) + ADD_K * vocab_size
        return num / den

    def evaluate(order: int, kn: bool) -> tuple[float, float]:
        log_prob = 0.0
        seen = 0
        unseen = 0
        for tokens in iter_tokens(TEST):
            m = [w if w in vocab else UNK for w in tokens]
            if order == 2:
                prev = m[0]
                for cur in m[1:]:
                    cont = bigram.get(prev)
                    if not cont or cur not in cont:
                        unseen += 1
                    p = prob2_kn(prev, cur) if kn else prob2_addk(prev, cur)
                    log_prob += math.log(p)
                    seen += 1
                    prev = cur
            else:
                w1, w2 = m[0], m[1]
                for cur in m[2:]:
                    ctx = trigram.get((w1, w2))
                    if not ctx or cur not in ctx:
                        unseen += 1
                    p = prob3_kn(w1, w2, cur) if kn else prob3_addk(w1, w2, cur)
                    log_prob += math.log(p)
                    seen += 1
                    w1, w2 = w2, cur
        return math.exp(-log_prob / seen), unseen / seen

    results = {}
    for label, order, kn in [
        ("bigram_add_k", 2, False),
        ("bigram_kneser_ney", 2, True),
        ("trigram_add_k", 3, False),
        ("trigram_kneser_ney", 3, True),
    ]:
        ppl, unseen = evaluate(order, kn)
        results[label] = {"perplexity": round(ppl, 2), "unseen_share": round(unseen, 4)}
        log(f"{label}: PP={ppl:.2f}, невиденных {unseen:.1%}", t0)

    payload = {
        "note": "разовое сравнение N=2 vs N=3, sandbox/lab1_trigram.py",
        "vocab_size": vocab_size,
        "bigram_types": bigram_types,
        "trigram_types": trigram_types,
        "discount_bigram": round(d2, 4),
        "discount_trigram": round(d3, 4),
        "results": results,
    }
    out = OUT
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\n| Модель | Перплексия | Невиденных n-грамм |")
    print("|---|---|---|")
    for label in results:
        r = results[label]
        print(f"| {label} | {r['perplexity']} | {r['unseen_share']:.1%} |")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()

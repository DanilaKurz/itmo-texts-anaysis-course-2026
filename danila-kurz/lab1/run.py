"""Лабораторная 1: очистка, нормализация, Левенштейн, биграммная модель.

Запуск из каталога ``text-analysis``::

    .venv/bin/python lab1/run.py

Скрипт последовательно проходит все шаги задания, сохраняет артефакты в
``artifacts/lab1`` и печатает сводку. Повторный запуск переиспользует
кэш очищенного корпуса.
"""

from __future__ import annotations

import heapq
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pymorphy3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.textkit.cleaning import clean_text, noise_report, tokenize, with_boundaries
from src.textkit.dataset import length_stats, load_lenta, stratified_split
from src.textkit.levenshtein import iter_close_pairs
from src.textkit.lm import BigramLM, iter_tokens
from src.textkit.normalize import lemmatize, stem

ART = ROOT / "artifacts" / "lab1"
ART.mkdir(parents=True, exist_ok=True)

# Собственный список, чтобы не зависеть от скачивания корпусов NLTK.
STOPWORDS = set(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
    по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
    если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
    никогда можно при наконец два об другой хоть после над больше тот через эти нас
    про всего них какая много разве три эту моя впрочем хорошо свою этой перед
    иногда лучше чуть том нельзя такой им более всегда конечно всю между""".split()
)

WORKERS = 8


# --------------------------------------------------------------------- utils
def log(step: str, start: float, **extra) -> float:
    elapsed = time.time() - start
    suffix = " ".join(f"{k}={v}" for k, v in extra.items())
    print(f"[{step}] {elapsed:6.1f} c  {suffix}", flush=True)
    return elapsed


def _init_worker() -> None:
    """Прогревает анализатор в каждом процессе до начала работы."""
    lemmatize("слово")
    stem("слово")


def _norm_shard(texts: list[str]) -> tuple[Counter, Counter]:
    """Считает частоты лемм и стемов для порции документов."""
    lemma_counts: Counter[str] = Counter()
    stem_counts: Counter[str] = Counter()
    for text in texts:
        for token in tokenize(text):
            lemma_counts[lemmatize(token)] += 1
            stem_counts[stem(token)] += 1
    return lemma_counts, stem_counts


def merge(counters) -> Counter:
    total: Counter[str] = Counter()
    for c in counters:
        total.update(c)
    return total


# --------------------------------------------------------------------- steps
def step_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = time.time()
    df = load_lenta("compact")
    train, test = stratified_split(df, test_size=0.2, seed=42)
    log("1. Загрузка и сплит", start, всего=len(df), train=len(train), test=len(test))
    return train, test


def step_eda(train: pd.DataFrame, metrics: dict) -> None:
    start = time.time()
    stats = length_stats(train)
    metrics["eda"] = stats
    metrics["classes"] = {k: int(v) for k, v in train["topic"].value_counts().items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    counts = train["topic"].value_counts()
    axes[0].bar(range(len(counts)), counts.values)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=30, ha="right")
    axes[0].set_title("Документы по темам")
    axes[0].set_ylabel("шт.")

    words = train["text"].str.split().str.len()
    axes[1].hist(words, bins=80, log=True)
    axes[1].set_xlim(0, 600)
    axes[1].set_title("Распределение длин (слова)")
    axes[1].set_xlabel("слов в документе")
    axes[1].set_ylabel("документов (лог)")

    tokens = Counter()
    for text in train["text"]:
        tokens.update(w.lower() for w in text.split())
    top_all = [(w, c) for w, c in tokens.most_common(300) if w.isalpha()][:30]
    stop_share = sum(c for w, c in tokens.items() if w in STOPWORDS) / sum(tokens.values())
    axes[2].barh([w for w, _ in reversed(top_all)], [c for _, c in reversed(top_all)])
    axes[2].set_title("Топ-30 частых токенов (почти все — стоп-слова)")
    axes[2].set_xlabel("частота")

    fig.tight_layout()
    fig.savefig(ART / "fig_eda.png", dpi=150)
    plt.close(fig)

    metrics["stopwords"] = {
        "share_of_tokens": round(stop_share, 4),
        "top30": [[w, int(c)] for w, c in top_all],
        "top30_content": [
            [w, int(c)] for w, c in tokens.most_common(400) if w not in STOPWORDS and w.isalpha()
        ][:30],
    }
    log("2. EDA", start, слов_в_среднем=round(stats["words_mean"], 1), доля_стопслов=round(stop_share, 3))


def step_noise(train: pd.DataFrame, metrics: dict) -> None:
    start = time.time()
    report = noise_report(train["text"].tolist())
    metrics["noise"] = {k: {"docs": v["docs"], "share": round(v["share"], 4)} for k, v in report.items()}
    pd.DataFrame(metrics["noise"]).T.to_csv(ART / "noise_stats.csv")

    examples = []
    for name, pattern in [("url", "http"), ("handle", "@"), ("hashtag", "#")]:
        sample = train.loc[train["text"].str.contains(pattern, regex=False), "text"].head(1)
        if len(sample):
            examples.append({"type": name, "text": sample.iloc[0][:300]})
    metrics["noise_examples"] = examples
    log("3. Анализ шума", start, типов=len(report))


def step_clean(
    train: pd.DataFrame, test: pd.DataFrame, metrics: dict
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    start = time.time()
    tokens_train = ART / "tokens_train.txt"
    tokens_test = ART / "tokens_test.txt"
    clean_train = ART / "clean_train.parquet"
    clean_test = ART / "clean_test.parquet"

    if tokens_train.exists() and tokens_test.exists() and clean_train.exists() and clean_test.exists():
        source = train["text"].iloc[0]
        metrics["cleaning_example"] = {"before": source[:400], "after": clean_text(source)[:400]}
        log("4. Очистка (из кэша)", start)
        return pd.read_parquet(clean_train), pd.read_parquet(clean_test), tokens_train, tokens_test

    cleaned = []
    for frame, clean_path, tokens_path in [
        (train, clean_train, tokens_train),
        (test, clean_test, tokens_test),
    ]:
        frame = frame.copy()
        frame["clean"] = frame["text"].map(clean_text)
        frame[["topic", "title", "clean"]].to_parquet(clean_path, index=False)
        with open(tokens_path, "w", encoding="utf-8") as handle:
            for text in frame["clean"]:
                handle.write(" ".join(with_boundaries(tokenize(text))) + "\n")
        cleaned.append(frame)

    source = cleaned[0]["text"].iloc[0]
    metrics["cleaning_example"] = {"before": source[:400], "after": clean_text(source)[:400]}
    log("4. Очистка и токенизация", start)
    return cleaned[0], cleaned[1], tokens_train, tokens_test


def step_normalize(train_clean: pd.DataFrame, metrics: dict) -> Counter:
    start = time.time()
    texts = train_clean["clean"].tolist()
    shards = [texts[i::WORKERS] for i in range(WORKERS)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as pool:
        results = list(pool.map(_norm_shard, shards))

    lemma_counts = merge(r[0] for r in results)
    stem_counts = merge(r[1] for r in results)
    raw_counts: Counter[str] = Counter()
    for text in texts:
        raw_counts.update(tokenize(text))

    metrics["vocab"] = {
        "raw": len(raw_counts),
        "stem": len(stem_counts),
        "lemma": len(lemma_counts),
        "raw_tokens": int(sum(raw_counts.values())),
        "stem_ratio": round(len(stem_counts) / len(raw_counts), 4),
        "lemma_ratio": round(len(lemma_counts) / len(raw_counts), 4),
    }
    metrics["vocab_examples"] = {
        "lemma": [f"{w} -> {lemmatize(w)}" for w, _ in raw_counts.most_common(200)][:12],
        "stem": [f"{w} -> {stem(w)}" for w, _ in raw_counts.most_common(200)][:12],
    }
    log(
        "5. Стемминг и лемматизация",
        start,
        raw=len(raw_counts),
        stem=len(stem_counts),
        lemma=len(lemma_counts),
    )
    return raw_counts


def step_levenshtein(raw_counts: Counter, metrics: dict) -> None:
    """Ищет пары на расстоянии 1–2 и разделяет их на три класса.

    Просто «топ-10 по частоте» бесполезен: в русском любые две словоформы
    одного слова отличаются на 1–2 правки, поэтому верх списка занимают
    «года/году/годы». Пары делятся на:

    * **словоформы** — одна лемма, обе частоты сопоставимы;
    * **ложные срабатывания** — разные леммы, обе частоты высокие
      («города/года», «кампании/компании»): строковая метрика их не различает;
    * **опечатки-кандидаты** — редкое слово неизвестно морфологическому
      словарю, а частое — известно. Именно так отсеиваются настоящие
      искажения от редких, но существующих слов.
    """
    start = time.time()
    min_freq = 10
    analyzer = pymorphy3.MorphAnalyzer()
    lemma_cache: dict[str, str] = {}
    known_cache: dict[str, bool] = {}

    def lemma(word: str) -> str:
        if word not in lemma_cache:
            lemma_cache[word] = analyzer.parse(word)[0].normal_form
        return lemma_cache[word]

    def known(word: str) -> bool:
        if word not in known_cache:
            known_cache[word] = analyzer.word_is_known(word)
        return known_cache[word]

    forms: list[tuple] = []
    near_miss: list[tuple] = []
    typos: list[tuple] = []
    total = d1 = d2 = 0

    for w1, w2, dist in iter_close_pairs(raw_counts, max_dist=2, min_freq=min_freq):
        total += 1
        d1 += dist == 1
        d2 += dist == 2
        f1, f2 = raw_counts[w1], raw_counts[w2]
        hi, lo = (w1, w2) if f1 >= f2 else (w2, w1)
        f_hi, f_lo = max(f1, f2), min(f1, f2)

        if f_lo >= 1000:
            item = (f_hi + f_lo, hi, lo, dist)
            target = forms if lemma(hi) == lemma(lo) else near_miss
            if len(target) < 20:
                heapq.heappush(target, item)
            elif item[0] > target[0][0]:
                heapq.heapreplace(target, item)
            continue

        if f_lo >= 2 and f_hi >= 200 and f_lo / f_hi <= 0.05 and not known(lo):
            item = (f_hi, lo, hi, dist, f_lo)
            if len(typos) < 40:
                heapq.heappush(typos, item)
            elif item[0] > typos[0][0]:
                heapq.heapreplace(typos, item)

    seen: set[str] = set()
    top_typos = []
    for f_hi, lo, hi, dist, f_lo in sorted(typos, reverse=True):
        if hi in seen:
            continue
        seen.add(hi)
        top_typos.append({"rare": lo, "likely": hi, "distance": dist, "freq_rare": f_lo, "freq_frequent": f_hi})
        if len(top_typos) == 10:
            break

    metrics["levenshtein"] = {
        "pairs_found": total,
        "distance1": d1,
        "distance2": d2,
        "vocab_checked": sum(1 for c in raw_counts.values() if c >= min_freq),
        "forms_top10": [
            {"a": a, "b": b, "distance": dist, "freq_total": f}
            for f, a, b, dist in sorted(forms, reverse=True)[:10]
        ],
        "near_miss_top10": [
            {"a": a, "b": b, "distance": dist, "freq_total": f}
            for f, a, b, dist in sorted(near_miss, reverse=True)[:10]
        ],
        "typos_top10": top_typos,
    }
    log("6. Левенштейн", start, пар=total, опечаток=len(top_typos))


def step_lm(tokens_train: Path, tokens_test: Path, metrics: dict) -> None:
    start = time.time()

    def train_iter():
        return iter_tokens(tokens_train)

    def test_iter():
        return iter_tokens(tokens_test)

    model = BigramLM(smoothing="kneser-ney", min_count=2).fit(train_iter)
    fit_time = time.time() - start

    unseen = model.unseen_bigram_share(test_iter)
    ppl_kn = model.perplexity(test_iter)
    ppl_addk = model.perplexity(test_iter, smoothing="add-k")

    metrics["lm"] = {
        "n": 2,
        "vocab_size": model.vocab_size,
        "bigram_types": sum(len(c) for c in model.bigram.values()),
        "bigrams_total": model.token_total,
        "discount": round(model.discount, 4),
        "unseen_bigram_share": round(unseen, 4),
        "perplexity_unsmoothed": None if unseen == 0 else "inf (есть биграммы с нулевой вероятностью)",
        "perplexity_kneser_ney": round(ppl_kn, 2),
        "perplexity_add_k": round(ppl_addk, 2),
        "fit_seconds": round(fit_time, 1),
    }
    temperatures = [0.5, 0.8, 0.8, 1.0, 1.2]
    metrics["generated"] = [
        {"temperature": t, "text": model.generate(max_len=40, temperature=t, seed=i)}
        for i, t in enumerate(temperatures)
    ]
    log(
        "7. Языковая модель",
        start,
        ppl_kn=round(ppl_kn, 1),
        ppl_addk=round(ppl_addk, 1),
        unseen=round(unseen, 3),
    )


def main() -> None:
    metrics: dict = {"dataset": "lenta-ru compact (5 тем)"}
    train, test = step_split()
    step_eda(train, metrics)
    step_noise(train, metrics)
    train_clean, _, tokens_train, tokens_test = step_clean(train, test, metrics)
    raw_counts = step_normalize(train_clean, metrics)
    step_levenshtein(raw_counts, metrics)
    step_lm(tokens_train, tokens_test, metrics)

    with open(ART / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"\nГотово. Артефакты в {ART}")


if __name__ == "__main__":
    main()

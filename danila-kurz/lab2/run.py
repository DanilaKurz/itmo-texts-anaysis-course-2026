"""Лабораторная 2: POS-разметка, деревья зависимостей, пары, анафора.

Запуск из каталога ``text-analysis``::

    .venv/bin/python lab2/run.py

POS-статистика считается по стратифицированной подвыборке 50 тыс. документов,
разбор зависимостей — по 20 тыс.: полный корпус в 165 тыс. документов занял бы
около 10 минут только на POS, а доли частей речи стабилизируются на порядок
раньше.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.textkit import plots
from src.textkit.morphology import (
    apply_resolution_to_span,
    count_pos,
    extract_pairs,
    load_nlp,
    resolve_anaphora,
)

ART = ROOT / "artifacts" / "lab2"
ART.mkdir(parents=True, exist_ok=True)
CLEAN = ROOT / "artifacts" / "lab1" / "clean_train.parquet"

POS_SAMPLE = 50_000
PARSE_SAMPLE = 20_000
POS_ORDER = [
    "NOUN", "VERB", "ADJ", "PROPN", "ADP", "PRON", "ADV",
    "CONJ", "CCONJ", "SCONJ", "NUM", "PART", "DET", "AUX",
]


def log(step: str, start: float, **extra) -> None:
    elapsed = time.time() - start
    suffix = " ".join(f"{k}={v}" for k, v in extra.items())
    print(f"[{step}] {elapsed:6.1f} c  {suffix}", flush=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Пропорциональная выборка по темам."""
    parts = []
    for _, group in df.groupby("topic"):
        take = max(1, round(n * len(group) / len(df)))
        parts.append(group.sample(min(take, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- POS
def step_pos(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    start = time.time()
    sample = stratified_sample(df, POS_SAMPLE)
    cache = ART / "pos_metrics.json"

    if cache.exists():
        metrics["pos"] = json.loads(cache.read_text(encoding="utf-8"))
        log("1. POS-разметка (из кэша)", start, документов=metrics["pos"]["sample_docs"])
        return sample

    nlp = load_nlp(parser=False)

    by_topic: dict[str, Counter] = {t: Counter() for t in sample["topic"].unique()}
    total: Counter = Counter()
    docs_with: Counter = Counter()

    for i, (doc, topic) in enumerate(zip(nlp.pipe(sample["clean"], batch_size=64, n_process=8), sample["topic"])):
        tokens, docs = count_pos(doc)
        total.update(tokens)
        docs_with.update(docs)
        by_topic[topic].update(tokens)
        if (i + 1) % 10000 == 0:
            print(f"    POS: {i + 1}/{len(sample)}", flush=True)

    share = {pos: round(c / sum(total.values()), 4) for pos, c in total.most_common()}
    metrics["pos"] = {
        "sample_docs": len(sample),
        "tokens": int(sum(total.values())),
        "counts": dict(total.most_common()),
        "share": share,
        "doc_share": {p: round(c / len(sample), 4) for p, c in docs_with.most_common()},
    }
    metrics["pos"]["by_topic_share"] = {
        topic: {p: round(c / sum(counter.values()), 4) for p, c in counter.most_common(12)}
        for topic, counter in by_topic.items()
    }
    cache.write_text(json.dumps(metrics["pos"], ensure_ascii=False), encoding="utf-8")
    log("1. POS-разметка", start, документов=len(sample), токенов=int(sum(total.values())))
    return sample


def step_pos_figures(metrics: dict) -> None:
    plots.plot_pos(metrics, ART / "fig_pos.png")


# --------------------------------------------------------------- дерево и пары
def step_parse(sample: pd.DataFrame, metrics: dict) -> None:
    start = time.time()
    cache = ART / "parse_metrics.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        metrics.update(cached)
        log("2. Деревья и пары (из кэша)", start, документов=cached["parse"]["docs"])
        return

    parse_sample = sample.sample(min(PARSE_SAMPLE, len(sample)), random_state=7).reset_index(drop=True)
    nlp = load_nlp(parser=True)

    sv_counter: Counter = Counter()
    an_counter: Counter = Counter()
    medium: list = []
    anaphora_examples: list[dict] = []
    errors: dict[str, list[dict]] = {"latin": [], "abbrev": [], "eto": []}

    abbrev = {"минфин", "фнс", "цб", "рф", "сми", "ввп", "оон", "сша", "еэс"}

    # Один процесс: замеры показали, что на macOS spaCy c n_process>1
    # замедляется в разы из-за spawn-старта воркеров (5527 ток/с в один поток
    # против 2296 ток/с в восемь).
    for i, doc in enumerate(nlp.pipe(parse_sample["clean"], batch_size=64, n_process=1)):
        sv, an = extract_pairs(doc)
        sv_counter.update(sv)
        an_counter.update(an)

        # Кандидаты в ошибки теггера: латиница, аббревиатуры, «это».
        if len(errors["latin"]) < 60:
            for token in doc:
                if token.is_alpha and token.text.isascii() and len(token.text) > 3:
                    errors["latin"].append(
                        {"token": token.text, "pos": token.pos_, "tag": token.tag_,
                         "lemma": token.lemma_, "ctx": doc[max(0, token.i - 4): token.i + 5].text}
                    )
                    break
        if len(errors["abbrev"]) < 60:
            for token in doc:
                if token.text.lower() in abbrev:
                    errors["abbrev"].append(
                        {"token": token.text, "pos": token.pos_, "tag": token.tag_,
                         "lemma": token.lemma_, "ctx": doc[max(0, token.i - 4): token.i + 5].text}
                    )
                    break
        if len(errors["eto"]) < 40:
            for token in doc:
                if token.text.lower() == "это":
                    errors["eto"].append(
                        {"token": token.text, "pos": token.pos_, "tag": token.tag_,
                         "lemma": token.lemma_, "ctx": doc[max(0, token.i - 4): token.i + 5].text}
                    )
                    break

        # Десять предложений средней длины для визуализации дерева.
        if len(medium) < 10:
            for sentence in doc.sents:
                size = len([t for t in sentence if not t.is_punct])
                if 12 <= size <= 20:
                    medium.append(sentence.as_doc())
                    break

        # Анафора: накапливаем уверенные разборы с короткой дистанцией и без
        # повторов пар «местоимение → антецедент».
        if len(anaphora_examples) < 12 and i % 5 == 0:
            seen_keys = {(r["pronoun_lemma"], r["antecedent_lemma"]) for r in anaphora_examples}
            for res in resolve_anaphora(doc):
                if len(anaphora_examples) >= 12:
                    break
                if not res["confident"] or res["distance"] > 30:
                    continue
                key = (res["pronoun_lemma"], res["antecedent_lemma"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                res["before"] = res["sentence"]
                sentence = list(doc.sents)[res["sentence_index"]]
                res["after"] = apply_resolution_to_span(sentence, [res])
                anaphora_examples.append(res)

        if (i + 1) % 5000 == 0:
            print(f"    parse: {i + 1}/{len(parse_sample)}", flush=True)

    metrics["pairs"] = {
        "subject_verb_top20": [[a, b, int(c)] for (a, b), c in sv_counter.most_common(20)],
        "noun_adj_top20": [[a, b, int(c)] for (a, b), c in an_counter.most_common(20)],
        "subject_verb_unique": len(sv_counter),
        "noun_adj_unique": len(an_counter),
        "subject_verb_total": int(sum(sv_counter.values())),
        "noun_adj_total": int(sum(an_counter.values())),
    }
    metrics["anaphora_examples"] = anaphora_examples[:8]
    metrics["tagger_errors"] = errors

    (ART / "tagger_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Деревья зависимостей.
    try:
        from spacy import displacy

        tree_dir = ART / "trees"
        tree_dir.mkdir(exist_ok=True)
        for idx, doc in enumerate(medium, start=1):
            svg = displacy.render(
                doc, style="dep", jupyter=False,
                options={"compact": True, "bg": "#ffffff", "distance": 110,
                         "font": "DejaVu Sans"},
            )
            (tree_dir / f"tree_{idx:02d}.svg").write_text(svg, encoding="utf-8")
        metrics["trees"] = [d.text for d in medium]
    except Exception as exc:  # noqa: BLE001 — визуализация не должна ронять пайплайн
        metrics["trees_error"] = str(exc)

    log(
        "2. Деревья и пары",
        start,
        документов=len(parse_sample),
        пар_sv=len(sv_counter),
        пар_an=len(an_counter),
        предложений=len(medium),
        анафор=len(anaphora_examples),
    )

    cache.write_text(
        json.dumps(
            {
                "pairs": metrics["pairs"],
                "anaphora_examples": metrics["anaphora_examples"],
                "tagger_errors": metrics["tagger_errors"],
                "trees": metrics["trees"],
                "parse": {"docs": len(parse_sample)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def step_figures(metrics: dict) -> None:
    plots.plot_pairs(metrics, ART / "fig_pairs.png")


def main() -> None:
    metrics: dict = {"dataset": "lenta-ru compact, очищенный корпус лабы 1"}
    df = pd.read_parquet(CLEAN)
    sample = step_pos(df, metrics)
    step_pos_figures(metrics)
    step_parse(sample, metrics)
    step_figures(metrics)

    (ART / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nГотово. Артефакты в {ART}")


if __name__ == "__main__":
    main()

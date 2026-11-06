"""Нормализация токенов: стемминг и лемматизация.

Лемматизация делается pymorphy3, стемминг — Snowball для русского. Оба
варианта применяются к одному и тому же набору токенов, чтобы сравнение было
честным: одинаковый корпус, одинаковый порог частоты, только разная обработка.
"""

from __future__ import annotations

import pymorphy3
from nltk.stem.snowball import SnowballStemmer

_ANALYZER: pymorphy3.MorphAnalyzer | None = None
_STEMMER: SnowballStemmer | None = None
# Кеш критичен для скорости: в корпусе 37 млн токенов и лишь ~600 тыс. типов,
# поэтому каждый разбор повторяется в среднем 60 раз.
_LEMMA_CACHE: dict[str, str] = {}
_STEM_CACHE: dict[str, str] = {}


def _lazy_init() -> tuple[pymorphy3.MorphAnalyzer, SnowballStemmer]:
    global _ANALYZER, _STEMMER
    if _ANALYZER is None:
        _ANALYZER = pymorphy3.MorphAnalyzer()
        _STEMMER = SnowballStemmer("russian")
    return _ANALYZER, _STEMMER


def stem(word: str) -> str:
    """Отрезает словоизменительные суффиксы алгоритмом Snowball."""
    cached = _STEM_CACHE.get(word)
    if cached is not None:
        return cached
    _, stemmer = _lazy_init()
    result = stemmer.stem(word)
    _STEM_CACHE[word] = result
    return result


def lemmatize(word: str) -> str:
    """Возвращает нормальную форму слова по наиболее вероятному разбору."""
    cached = _LEMMA_CACHE.get(word)
    if cached is not None:
        return cached
    analyzer, _ = _lazy_init()
    result = analyzer.parse(word)[0].normal_form
    _LEMMA_CACHE[word] = result
    return result


def normalize_counts(
    token_counter,
    vocab: set[str] | None = None,
    method: str = "lemma",
) -> dict[str, int]:
    """Схлопывает частоты токенов по стемам или леммам.

    Parameters:
        token_counter (Mapping[str, int]): частоты исходных токенов.
        vocab (set[str] | None): если задан, обрабатываются только эти слова.
        method (str): ``lemma`` или ``stem``.

    Returns:
        dict[str, int]: частоты нормализованных форм.

    Примечание:
        Внутри держится кеш «слово → форма»: словарь корпуса на два порядка
        меньше числа токенов, поэтому повторные разборы не выполняются.
    """
    transform = lemmatize if method == "lemma" else stem
    cache: dict[str, str] = {}
    out: dict[str, int] = {}
    for word, count in token_counter.items():
        if vocab is not None and word not in vocab:
            continue
        form = cache.get(word)
        if form is None:
            form = transform(word)
            cache[word] = form
        out[form] = out.get(form, 0) + count
    return out

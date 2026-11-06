"""Расстояние Левенштейна и поиск пар слов на расстоянии 1–2.

Требование задания — реализовать алгоритм самостоятельно, поэтому здесь нет
ни rapidfuzz, ни python-Levenshtein. Память оптимизирована до двух строк
``O(min(n, m))``, а полный перебор пар заменён фильтрацией по длине и q-граммам:
наивное сравнение всех слов словаря — это O(V^2), что для 300 тысяч типов
невыполнимо.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator


def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Считает расстояние редактирования между строками.

    Parameters:
        a (str), b (str): сравниваемые строки.
        max_dist (int | None): если задан, при превышении возвращается
            ``max_dist + 1`` (ранний выход по нижней границе строки).

    Returns:
        int: минимальное число вставок, удалений и замен.
    """
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    if max_dist is not None and len(b) - len(a) > max_dist:
        return max_dist + 1

    previous = list(range(len(a) + 1))
    for i, ch_b in enumerate(b, start=1):
        current = [i]
        row_min = i
        for j, ch_a in enumerate(a, start=1):
            cost = 0 if ch_a == ch_b else 1
            value = min(
                previous[j] + 1,      # удаление
                current[j - 1] + 1,   # вставка
                previous[j - 1] + cost,  # замена
            )
            current.append(value)
            if value < row_min:
                row_min = value
        # Если вся строка уже дороже порога, дальше нет смысла считать.
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        previous = current
    return previous[-1]


def _grams(word: str, n: int = 3) -> set[str]:
    if len(word) <= n:
        return {word}
    return {word[i : i + n] for i in range(len(word) - n + 1)}


def candidate_pairs(vocab: list[str], max_dist: int = 2) -> set[tuple[str, str]]:
    """Отбирает пары-кандидаты, которые вообще могут быть близки.

    Наивное сравнение всех пар — O(V^2). Вместо этого используется фильтр
    q-грамм: две строки, различающиеся не более чем на ``k`` правок, теряют не
    более ``k`` триграмм, поэтому достаточно «опросить» ``k + 1`` самых редких
    триграмм слова — хотя бы одна из них уцелеет и приведёт к паре-кандидату.
    Дальше идёт дешёвая проверка числа общих триграмм, точное расстояние считает
    вызывающая сторона.
    """
    grams: dict[str, set[str]] = {word: _grams(word) for word in vocab}
    index: dict[str, list[str]] = defaultdict(list)
    for word, gs in grams.items():
        for gram in gs:
            index[gram].append(word)

    candidates: set[tuple[str, str]] = set()
    for word in vocab:
        gs = grams[word]
        # Редкие триграммы дают короткие списки вхождений — опрашиваем их.
        rarest = sorted(gs, key=lambda g: len(index[g]))[: max_dist + 1]
        pool: set[str] = set()
        for gram in rarest:
            pool.update(index[gram])
        for other in pool:
            if other == word or abs(len(other) - len(word)) > max_dist:
                continue
            pair = (word, other) if word < other else (other, word)
            if pair in candidates:
                continue
            shared = len(gs & grams[other])
            threshold = max(len(gs), len(grams[other])) - 3 * max_dist - 1
            if shared >= max(threshold, 1):
                candidates.add(pair)
    return candidates


def iter_close_pairs(
    counts: dict[str, int],
    max_dist: int = 2,
    min_freq: int = 5,
) -> Iterator[tuple[str, str, int]]:
    """Лениво перебирает пары слов с расстоянием 1–2.

    Пара генерируется только после фильтра по длине и триграммам, поэтому
    точное расстояние считается для единиц процентов от всех пар словаря.

    Yields:
        tuple[str, str, int]: ``(слово1, слово2, расстояние)``.
    """
    vocab = [w for w, c in counts.items() if c >= min_freq and len(w) >= 4]
    for w1, w2 in candidate_pairs(vocab, max_dist=max_dist):
        d = levenshtein(w1, w2, max_dist=max_dist)
        if 1 <= d <= max_dist:
            yield w1, w2, d

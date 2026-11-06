"""Обнаружение шума и конвейер очистки текста.

Шум ищется теми же регулярными выражениями, которыми потом чистится: так
таблица «тип шума → сколько документов» и «тип шума → правило» гарантированно
согласованы между собой.
"""

from __future__ import annotations

import html
import re
from collections import OrderedDict

# Порядок правил важен: сущности декодируются до удаления тегов, ссылки
# вырезаются до нормализации пробелов, числа заменяются последними.
RULES: "OrderedDict[str, tuple[str, str]]" = OrderedDict(
    [
        ("url", (r"https?://\S+|www\.\S+|\b[\w-]+\.(?:ru|com|org|net|info|io)\b\S*", " [URL] ")),
        ("email", (r"[\w.+-]+@[\w-]+\.[\w.]{2,}", " [EMAIL] ")),
        ("handle", (r"(?<!\w)@[A-Za-z0-9_]{2,}", " [HANDLE] ")),
        ("hashtag", (r"(?<!\w)#[A-Za-zА-Яа-яЁё0-9_]{2,}", " [HASHTAG] ")),
        ("html_tag", (r"</?[a-zA-Z][^>]{0,40}>", " ")),
        ("html_entity", (r"&(?:[a-zA-Z]+|#\d+);", " ")),
        ("invisible_space", ("[\u00a0\u2007\u202f\u200b-\u200f\u2060\ufeff]", " ")),
        ("emoji", ("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]", " ")),
        ("censored", (r"\*{2,}", " ")),
        ("number", (r"\d+(?:[.,]\d+)*", " [NUM] ")),
        ("quote", (r"[«»„“”]", '"')),
        ("dash", (r"[–—−]", "-")),
        ("multi_space", (r"[ \t]{2,}", " ")),
        ("newline", (r"\s*\n+\s*", " ")),
        ("scraper_artifact", (r"^\s*\.?RedirectTo\s*", " ")),
        ("word_glue", (r"\s+([,.;:!?])", r"\1")),
    ]
)

_COMPILED = OrderedDict((name, (re.compile(pattern), repl)) for name, (pattern, repl) in RULES.items())


def _count_pattern(texts, pattern: re.Pattern) -> int:
    return sum(1 for t in texts if pattern.search(t))


def noise_report(texts) -> "OrderedDict[str, dict]":
    """Возвращает по каждому типу шума число документов, где он встретился.

    Parameters:
        texts (Iterable[str]): исходные тексты.

    Returns:
        OrderedDict[str, dict]: ``{"docs": int, "share": float}`` по типам.
    """
    texts = list(texts)
    total = len(texts)
    report: "OrderedDict[str, dict]" = OrderedDict()
    for name, (pattern, _) in _COMPILED.items():
        hits = _count_pattern(texts, pattern)
        report[name] = {"docs": hits, "share": hits / total if total else 0.0}
    return report


def clean_text(text: str) -> str:
    """Применяет каскад правил и возвращает очищенный текст.

    Удаляется только то, что не несёт смысла (теги, невидимые символы);
    значимые сущности заменяются токенами ``[URL]``, ``[NUM]`` и т.п., чтобы
    модель могла выучить их как отдельные единицы.
    """
    out = html.unescape(text)
    for pattern, repl in _COMPILED.values():
        out = pattern.sub(repl, out)
    return out.strip()


TOKEN_RE = re.compile(r"\[[A-Z]+\]|[а-яёa-z]+", re.IGNORECASE)
BOUNDARY_START = "<s>"
BOUNDARY_END = "</s>"


def tokenize(text: str) -> list[str]:
    """Режет очищенный текст на токены: слова и ``[URL]``-подобные метки.

    Служебные метки приводятся к верхнему регистру (``[URL]``), обычные слова —
    к нижнему. Пунктуация отбрасывается осознанно: она не нужна ни для
    словаря, ни для n-граммной модели, а её обработку задание не требует.
    """
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text):
        tokens.append(match.upper() if match.startswith("[") else match.lower())
    return tokens


def with_boundaries(tokens: list[str]) -> list[str]:
    """Оборачивает последовательность токенов маркерами ``<s>`` и ``</s>``."""
    return [BOUNDARY_START, *tokens, BOUNDARY_END]

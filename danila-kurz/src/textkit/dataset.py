"""Загрузка датасета Lenta.ru и стратифицированное разбиение.

Доступны две версии корпуса, обе лежат в ``data/lenta-ru-news``:

* ``compact`` — 5 крупных тем, колонки ``title/text/topic/labels``;
* ``full`` — полный корпус, 10+ тем, есть ``date``, ``url`` и ``tags``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "lenta-ru-news"


def load_lenta(kind: str = "compact") -> pd.DataFrame:
    """Читает корпус, убирает дубли и пустые тексты.

    Parameters:
        kind (str): ``compact`` или ``full``.

    Returns:
        pd.DataFrame: колонки ``title``, ``text``, ``topic``; для ``full``
        дополнительно ``date`` и ``tags``.
    """
    if kind == "compact":
        frames = [
            pd.read_parquet(DATA_DIR / "train.parquet"),
            pd.read_parquet(DATA_DIR / "test.parquet"),
        ]
        df = pd.concat(frames, ignore_index=True)
        df = df[["title", "text", "topic"]]
    elif kind == "full":
        df = pd.read_parquet(DATA_DIR / "full.parquet")
        df = df[df["topic"].notna() & df["text"].notna()]
        df = df[["title", "text", "topic", "date", "tags"]]
    else:
        raise ValueError(f"неизвестный вариант датасета: {kind}")

    # Часть статей пришла из скрейпера с пустым или вырожденным текстом.
    df = df.dropna(subset=["text", "topic"])
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    label: str = "topic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит корпус на train/test с сохранением долей классов."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def length_stats(df: pd.DataFrame, column: str = "text") -> dict:
    """Считает распределение длин в словах и символах."""
    chars = df[column].str.len()
    words = df[column].str.split().str.len()
    return {
        "docs": int(len(df)),
        "chars_mean": float(chars.mean()),
        "chars_median": float(chars.median()),
        "chars_p90": float(chars.quantile(0.9)),
        "chars_p99": float(chars.quantile(0.99)),
        "chars_max": int(chars.max()),
        "words_mean": float(words.mean()),
        "words_median": float(words.median()),
        "words_p90": float(words.quantile(0.9)),
        "words_max": int(words.max()),
        "words_total": int(words.sum()),
    }

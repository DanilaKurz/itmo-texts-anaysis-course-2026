"""Функции отрисовки всех графиков курса в едином стиле.

Каждая функция принимает словарь метрик (как в ``artifacts/lab*/metrics.json``)
и путь для сохранения PNG. Это позволяет перерисовывать графики без повторного
запуска пайплайнов: сценарий ``scripts/make_figures.py`` делает ровно это.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.textkit.style import (
    BLUE_RAMP,
    C_BLUE,
    C_CORAL,
    C_LAV,
    C_PEACH,
    C_SAND,
    C_TEAL,
    GRID,
    INK,
    MUTED,
    PALETTE,
    caption,
    frame,
    hbars,
    new_figure,
    vbars,
)

# ---------------------------------------------------------------- лаба 1

def plot_eda(metrics: dict, df: pd.DataFrame | None, out: Path) -> None:
    """Три панели: классы, длины документов, топ-30 токенов."""
    fig = new_figure(11.5, 4.3)

    # Классы.
    ax1 = fig.add_subplot(1, 3, 1)
    topics = list(metrics["classes"].keys())
    values = list(metrics["classes"].values())
    ax1.set_title("Документы по темам", pad=12)
    hbars(ax1, topics, values, colors=PALETTE[: len(topics)])

    # Длины в словах.
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Длина документов, слов", pad=12)
    lengths = df["text"].str.split().str.len() if df is not None else []
    counts, edges = np.histogram(lengths, bins=60, range=(0, 600))
    width = edges[1] - edges[0]
    for edge, count in zip(edges[:-1], counts):
        if count <= 0:
            continue
        patch = FancyBboxPatch(
            (edge, 0), width * 0.92, count,
            boxstyle="round,pad=0,rounding_size=2.2",
            mutation_scale=1.0,
            facecolor=C_BLUE,
            edgecolor="none",
            alpha=0.9,
        )
        ax2.add_patch(patch)
    frame(ax2)
    ax2.set_yscale("log")
    ax2.set_ylim(0.9, None)
    ax2.set_xlim(0, 620)
    ax2.set_xlabel("слов в документе")
    ax2.set_ylabel("документов (лог)")
    median = float(lengths.median()) if len(lengths) else 0
    ax2.axvline(median, color=C_PEACH, linewidth=1.6, linestyle="--")
    ax2.text(median + 12, ax2.get_ylim()[1] * 0.75, f"медиана {median:.0f}",
             color=C_PEACH, fontsize=9, fontweight="bold")

    # Топ-30 токенов.
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Топ-30 токенов", pad=12)
    rows = metrics["stopwords"]["top30"][:30]
    hbars(ax3, [w for w, _ in reversed(rows)], [c for _, c in reversed(rows)],
          color=C_PEACH)

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Lenta.ru News Dataset, обучающая выборка (165 143 документа)")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_noise(metrics: dict, out: Path) -> None:
    """Доля документов с каждым типом шума, лог-шкала."""
    systemic = {"number", "quote", "dash", "multi_space", "url",
                "invisible_space", "newline"}
    noise = metrics["noise"]
    rows = sorted(noise.items(), key=lambda kv: -kv[1]["share"])

    fig = new_figure(11.5, 4.3)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Шум в корпусе: доля документов с каждым типом", pad=12)
    labels = [name for name, _ in rows]
    values = [v["share"] * 100 for _, v in rows]
    colors = [C_BLUE if name in systemic else C_SAND for name in labels]
    hbars(ax, labels, values, colors=colors, log=True, value_fmt="{:.2f}%",
          value_offset_frac=0.12)
    ax.set_xlabel("% документов (лог-шкала)")
    from matplotlib.patches import Patch
    # Легенда — над графиком, в две колонки: внутри осей она наезжает на бары.
    ax.legend(
        handles=[Patch(color=C_BLUE, label="системный шум"),
                 Patch(color=C_SAND, label="редкие типы")],
        loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2,
        frameon=False, fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    caption(fig, "Измерено на train теми же регулярками, которыми ведётся очистка")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_vocab(metrics: dict, out: Path) -> None:
    """Размер словаря до и после нормализации."""
    vocab = metrics["vocab"]
    fig = new_figure(7.6, 4.0)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Словарь до и после нормализации", pad=12)
    vbars(
        ax,
        ["без нормализации", "стемминг", "лемматизация"],
        [vocab["raw"], vocab["stem"], vocab["lemma"]],
        colors=[C_BLUE, C_TEAL, C_PEACH],
    )
    ax.set_ylabel("уникальных токенов")
    # Проценты — внутри баров, чтобы не спорить с числами над ними.
    # Тёмный текст: белый на светлых пастельных барах теряет контраст.
    for i, key in enumerate(("stem", "lemma")):
        ax.text(
            i + 1, vocab[key] * 0.42,
            f"−{round((1 - vocab[f'{key}_ratio']) * 100)} %",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK,
        )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Один и тот же корпус train, порог частоты одинаковый")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_ppl(metrics: dict, out: Path) -> None:
    """Перплексия на тесте: без сглаживания и с двумя методами."""
    lm = metrics["lm"]
    unseen = lm["unseen_bigram_share"]
    fig = new_figure(8.4, 4.0)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Перплексия биграммной модели на тесте", pad=12)

    bars = [("Kneser-Ney", lm["perplexity_kneser_ney"], C_BLUE),
            ("Add-k (k=0.01)", lm["perplexity_add_k"], C_PEACH)]
    for i, (name, value, color) in enumerate(bars):
        patch = FancyBboxPatch(
            (0, i - 0.3), value, 0.6,
            boxstyle="round,pad=0,rounding_size=0.3",
            mutation_scale=1.0, facecolor=color, edgecolor="none", alpha=0.95,
        )
        ax.add_patch(patch)
        ax.text(value + 22, i, f"{value:.0f}", va="center", ha="left",
                fontsize=9, color=INK, fontweight="bold")

    patch = FancyBboxPatch(
        (0, 2 - 0.3), lm["perplexity_add_k"] * 0.4, 0.6,
        boxstyle="round,pad=0,rounding_size=0.3",
        mutation_scale=1.0, facecolor="none", edgecolor=C_CORAL,
        linewidth=1.4, linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(lm["perplexity_add_k"] * 0.4 + 22, 2, "∞", va="center", ha="left",
            fontsize=11, color=C_CORAL, fontweight="bold")
    ax.text(0, 2.62, f"без сглаживания: {unseen * 100:.1f} % тестовых биграмм "
                     f"имеют нулевую вероятность, поэтому перплексия не определена",
            va="bottom", fontsize=8.8, color=C_CORAL)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([n for n, _, _ in bars] + ["без сглаживания"])
    frame(ax)
    ax.set_ylim(-0.75, 3.05)
    ax.set_xlim(0, lm["perplexity_add_k"] * 1.45)
    ax.set_xlabel("перплексия (меньше — лучше)")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- лаба 2

def plot_pos(metrics: dict, out: Path) -> None:
    """Доля частей речи в корпусе и по темам."""
    pos = metrics["pos"]
    order = [p for p, _ in sorted(pos["share"].items(), key=lambda kv: -kv[1])]
    order = order[:9]

    fig = new_figure(11.5, 4.6)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Части речи в корпусе", pad=12)
    hbars(ax1, [p for p in reversed(order)],
          [pos["share"][p] * 100 for p in reversed(order)],
          color=C_BLUE, value_fmt="{:.1f}%")
    ax1.set_xlabel("% токенов")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Доля частей речи по темам", pad=12)
    topics = sorted(pos["by_topic_share"])
    matrix = [[pos["by_topic_share"][t].get(p, 0) * 100 for p in order]
              for t in topics]
    image = ax2.imshow(matrix, aspect="auto", cmap=BLUE_RAMP)
    ax2.set_xticks(range(len(order)), order, rotation=40, ha="right", fontsize=8.5)
    ax2.set_yticks(range(len(topics)), topics, fontsize=9)
    for i in range(len(topics)):
        for j in range(len(order)):
            value = matrix[i][j]
            ax2.text(j, i, f"{value:.1f}", ha="center", va="center",
                     fontsize=7.6, color=INK if value < 14 else "white")
    for side in ("top", "right", "bottom", "left"):
        ax2.spines[side].set_visible(False)
    ax2.tick_params(length=0)
    fig.colorbar(image, ax=ax2, shrink=0.75, pad=0.02)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Подвыборка 50 000 документов, 9.1 млн токенов; доли в %")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pairs(metrics: dict, out: Path) -> None:
    """Топ-12 пар обоих типов."""
    pairs = metrics["pairs"]
    fig = new_figure(11.5, 4.6)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("«Субъект + действие»", pad=12)
    rows = pairs["subject_verb_top20"][:12]
    hbars(ax1, [f"{a} — {b}" for a, b, _ in reversed(rows)],
          [c for _, _, c in reversed(rows)], color=C_TEAL)
    ax1.set_xlabel("частота в 20 тыс. документов")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("«Существительное + прилагательное»", pad=12)
    rows = pairs["noun_adj_top20"][:12]
    hbars(ax2, [f"{a} — {b}" for a, b, _ in reversed(rows)],
          [c for _, _, c in reversed(rows)], color=C_LAV)
    ax2.set_xlabel("частота в 20 тыс. документов")

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Пары извлечены обходом дерева зависимостей (nsubj и amod)")
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- все сразу
ALL = {
    "lab1": [("fig_eda", plot_eda), ("fig_noise", plot_noise),
             ("fig_vocab", plot_vocab), ("fig_ppl", plot_ppl)],
    "lab2": [("fig_pos", plot_pos), ("fig_pairs", plot_pairs)],
}

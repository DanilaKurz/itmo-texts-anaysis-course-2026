"""Единый визуальный стиль графиков: мягкая палитра, скруглённые бары.

Все рисунки курса рисуются через этот модуль, чтобы отчёты выглядели
согласованно: светло-тёплый фон, приглушённые цвета, скруглённые
«стадионные» бары, горизонтальная сетка, подписи значений прямо на барах.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------- палитра
INK = "#33302E"      # тёмно-графитовый текст
PAPER = "#FAF7F2"    # тёплый фон
GRID = "#E8E2D8"     # линии сетки
MUTED = "#8F887E"    # подписи осей
SOFT = "#C7C0B5"     # приглушённый акцент

C_BLUE = "#5B8DEF"
C_PEACH = "#F2A65A"
C_TEAL = "#63C5B5"
C_LAV = "#A78BE0"
C_CORAL = "#E8736B"
C_SAND = "#D9C9A8"

PALETTE = [C_BLUE, C_PEACH, C_TEAL, C_LAV, C_CORAL, C_SAND]

BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#F2F6FD", "#5B8DEF"])


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "figure.dpi": 140,
        }
    )


def new_figure(width: float, height: float):
    """Создаёт рисунок на тёплом фоне с заданным размером в дюймах."""
    _setup()
    fig = plt.figure(figsize=(width, height), facecolor=PAPER)
    return fig


def frame(ax) -> None:
    """Убирает лишние границы, оставляет лёгкую сетку по Y."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1, alpha=0.9)
    ax.xaxis.grid(False)


def hbars(ax, labels: list[str], values: list[float], color=None, colors=None,
          value_fmt: str = "{:,.0f}", height: float = 0.62, log: bool = False,
          value_offset_frac: float = 0.015) -> None:
    """Горизонтальные скруглённые бары с подписями значений.

    Parameters:
        ax: оси.
        labels, values: подписи и значения сверху вниз.
        color: один цвет для всех баров.
        colors: список цветов по барам (переопределяет ``color``).
        value_fmt: формат подписи значения.
        height: высота бара в долях шага.
        log: логарифмическая шкала X (для сильно разных значений).
        value_offset_frac: отступ подписи от конца бара в долях диапазона.

    Границы осей выставляются до отрисовки подписей: иначе текст с большими
    значениями оказывается на миллионы пикселей за пределами осей, и FreeType
    падает с «raster overflow».
    """
    n = len(labels)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    frame(ax)

    vmax = max(values)
    if log:
        vmin = min(v for v in values if v > 0) / 2
        ax.set_xscale("log")
        ax.set_xlim(vmin, vmax * 2.5)
    else:
        ax.set_xlim(0, vmax * 1.15)

    for i, (label, value) in enumerate(zip(labels, values)):
        if colors is not None:
            bar_color = colors[i]
        else:
            bar_color = color or C_BLUE
        patch = FancyBboxPatch(
            (0, i - height / 2), value, height,
            boxstyle=f"round,pad=0,rounding_size={height / 2}",
            mutation_scale=1.0,
            facecolor=bar_color,
            edgecolor="none",
            alpha=0.95,
            linewidth=0,
        )
        ax.add_patch(patch)
        label_x = value * 1.12 if log else value + vmax * value_offset_frac
        ax.text(label_x, i, value_fmt.format(value),
                va="center", ha="left", fontsize=8.5, color=INK)


def vbars(ax, labels: list[str], values: list[float], colors=None,
          value_fmt: str = "{:,.0f}", width: float = 0.6) -> None:
    """Вертикальные скруглённые бары с подписями значений над ними."""
    n = len(labels)
    ax.set_xlim(-0.75, n - 0.25)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    frame(ax)

    vmax = max(values)
    ax.set_ylim(0, vmax * 1.15)

    for i, (label, value) in enumerate(zip(labels, values)):
        bar_color = (colors[i] if colors is not None else C_BLUE)
        patch = FancyBboxPatch(
            (i - width / 2, 0), width, value,
            boxstyle=f"round,pad=0,rounding_size={width / 4}",
            mutation_scale=1.0,
            facecolor=bar_color,
            edgecolor="none",
            alpha=0.95,
            linewidth=0,
        )
        ax.add_patch(patch)
        ax.text(i, value + vmax * 0.02, value_fmt.format(value),
                va="bottom", ha="center", fontsize=8.5, color=INK)


def caption(fig, text: str) -> None:
    """Подпись под рисунком мелким приглушённым шрифтом."""
    fig.text(0.5, 0.015, text, ha="center", va="bottom",
             fontsize=8.5, color=MUTED)

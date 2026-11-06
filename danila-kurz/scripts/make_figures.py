"""Перерисовывает все графики курса из готовых metrics.json.

Запуск из каталога ``text-analysis``::

    .venv/bin/python scripts/make_figures.py

Пайплайны лаб не перезапускаются: рисунки собираются из уже посчитанных метрик,
поэтому перерисовка занимает секунды.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.textkit.dataset import load_lenta, stratified_split  # noqa: E402
from src.textkit.plots import ALL  # noqa: E402

ART1 = ROOT / "artifacts" / "lab1"
ART2 = ROOT / "artifacts" / "lab2"


def main() -> None:
    m1 = json.loads((ART1 / "metrics.json").read_text(encoding="utf-8"))
    m2 = json.loads((ART2 / "metrics.json").read_text(encoding="utf-8"))

    # Для гистограммы длин нужен train — восстанавливаем его тем же сплитом.
    df = load_lenta("compact")
    train, _ = stratified_split(df, test_size=0.2, seed=42)

    for name, fn in ALL["lab1"]:
        target = ART1 / f"{name}.png"
        fn(m1, train, target) if name == "fig_eda" else fn(m1, target)
        print(f"lab1/{target.name}")

    for name, fn in ALL["lab2"]:
        target = ART2 / f"{name}.png"
        fn(m2, target)
        print(f"lab2/{target.name}")


if __name__ == "__main__":
    main()

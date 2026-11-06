"""Биграммная языковая модель со сглаживанием и оценкой перплексии.

Реализованы два классических сглаживания:

* аддитивное (Add-k, вариант Лапласа при ``k = 1``);
* интерполированный Kneser-Ney с абсолютным дисконтированием.

Модель работает в два прохода по корпусу: первый собирает частоты слов и
определяет словарь (редкие слова уходят в ``[UNK]``), второй — биграммы уже
поверх словаря. Корпус подаётся через фабрику итераторов, чтобы не держать в
памяти 37 млн строковых объектов.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Callable, Iterable, Iterator

START = "<s>"
END = "</s>"
UNK = "[UNK]"


def iter_tokens(path) -> Iterator[list[str]]:
    """Читает файл, где один документ — одна строка токенов."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if tokens:
                yield tokens


class BigramLM:
    """Биграммная модель с выбором сглаживания.

    Parameters:
        smoothing (str): ``kneser-ney`` или ``add-k``.
        min_count (int): минимальная частота слова для попадания в словарь.
        add_k (float): параметр аддитивного сглаживания.
    """

    def __init__(self, smoothing: str = "kneser-ney", min_count: int = 2, add_k: float = 0.01) -> None:
        self.smoothing = smoothing
        self.min_count = min_count
        self.add_k = add_k
        self.vocab: set[str] = set()
        self.unigram: Counter[str] = Counter()
        self.bigram: dict[str, Counter[str]] = {}
        self.left_total: dict[str, int] = {}
        self.continuations: Counter[str] = Counter()
        self.left_types: dict[str, int] = {}
        self.discount = 0.0
        self.p_cont: dict[str, float] = {}
        self.vocab_size = 0
        self.token_total = 0

    # ------------------------------------------------------------------ fit
    def fit(self, make_iter: Callable[[], Iterable[list[str]]]) -> "BigramLM":
        """Обучает модель.

        Parameters:
            make_iter (Callable): функция без аргументов, возвращающая новый
                итератор по предложениям (спискам токенов). Нужна потому, что
                проходов два: словарь определяется до подсчёта биграмм.
        """
        counts: Counter[str] = Counter()
        for tokens in make_iter():
            counts.update(tokens)

        self.vocab = {w for w, c in counts.items() if c >= self.min_count}
        self.vocab |= {START, END, UNK}
        self.unigram = Counter({w: c for w, c in counts.items() if w in self.vocab})
        self.vocab_size = len(self.vocab)
        # Частоты UNK складываются из всех выброшенных слов: иначе вероятность
        # UNK окажется заниженной и перплексия на тесте «обрадуется» зря.
        self.unigram[UNK] = sum(c for w, c in counts.items() if w not in self.vocab)

        bigram: dict[str, Counter[str]] = defaultdict(Counter)
        for tokens in make_iter():
            mapped = [w if w in self.vocab else UNK for w in tokens]
            prev = mapped[0]
            for cur in mapped[1:]:
                bigram[prev][cur] += 1
                prev = cur

        self.bigram = dict(bigram)
        self.left_total = {w: sum(c.values()) for w, c in self.bigram.items()}
        self.left_types = {w: len(c) for w, c in self.bigram.items()}
        for w, cont in self.bigram.items():
            for nxt in cont:
                self.continuations[nxt] += 1
        self.token_total = sum(self.left_total.values())

        cont_total = sum(self.continuations.values())
        self.p_cont = {w: c / cont_total for w, c in self.continuations.items()}

        n1 = sum(1 for c in self.bigram.values() for v in c.values() if v == 1)
        n2 = sum(1 for c in self.bigram.values() for v in c.values() if v == 2)
        self.discount = n1 / (n1 + 2 * n2) if n1 and n2 else 0.5
        return self

    # ------------------------------------------------------------ probability
    def prob(self, w1: str, w2: str, smoothing: str | None = None) -> float:
        """Вероятность ``P(w2 | w1)`` при выбранном сглаживании."""
        smoothing = smoothing or self.smoothing
        if w1 not in self.vocab:
            w1 = UNK
        if w2 not in self.vocab:
            w2 = UNK
        cont = self.bigram.get(w1)
        if smoothing == "add-k":
            num = (cont[w2] if cont else 0) + self.add_k
            den = self.left_total.get(w1, 0) + self.add_k * self.vocab_size
            return num / den
        if smoothing == "kneser-ney":
            if not cont or w1 not in self.left_total:
                return self.p_cont.get(w2, 1e-12)
            observed = cont[w2]
            base = max(observed - self.discount, 0.0) / self.left_total[w1]
            backoff = self.discount * self.left_types[w1] / self.left_total[w1]
            return base + backoff * self.p_cont.get(w2, 1e-12)
        raise ValueError(f"неизвестное сглаживание: {smoothing}")

    def unseen_bigram_share(self, make_iter: Callable[[], Iterable[list[str]]]) -> float:
        """Доля биграмм теста, не встречавшихся в обучении."""
        total = unseen = 0
        for tokens in make_iter():
            mapped = [w if w in self.vocab else UNK for w in tokens]
            prev = mapped[0]
            for cur in mapped[1:]:
                total += 1
                cont = self.bigram.get(prev)
                if not cont or cur not in cont:
                    unseen += 1
                prev = cur
        return unseen / total if total else 0.0

    # ------------------------------------------------------------ perplexity
    def perplexity(
        self,
        make_iter: Callable[[], Iterable[list[str]]],
        smoothing: str | None = None,
    ) -> float:
        """Считает перплексию через сумму логарифмов правдоподобия.

        Логарифмы нужны, чтобы произведение вероятностей по 8 млн биграмм не
        ушло в машинный ноль.
        """
        log_prob = 0.0
        tokens_seen = 0
        for tokens in make_iter():
            mapped = [w if w in self.vocab else UNK for w in tokens]
            prev = mapped[0]
            for cur in mapped[1:]:
                p = self.prob(prev, cur, smoothing=smoothing)
                log_prob += math.log(p)
                tokens_seen += 1
                prev = cur
        if not tokens_seen:
            return float("inf")
        return math.exp(-log_prob / tokens_seen)

    # ----------------------------------------------------------- generation
    def _sample_next(self, w1: str, temperature: float, rng: random.Random) -> str:
        cont = self.bigram.get(w1)
        candidates: list[tuple[str, float]] = []
        backoff_mass = 1.0
        if cont:
            total = self.left_total[w1]
            for w2, c in cont.items():
                p = max(c - self.discount, 0.0) / total
                if p > 0 and w2 != UNK:
                    candidates.append((w2, p))
            backoff_mass = self.discount * self.left_types[w1] / total
        else:
            backoff_mass = 1.0

        if temperature != 1.0 and temperature > 0 and candidates:
            candidates = [(w, p ** (1.0 / temperature)) for w, p in candidates]
            norm = sum(p for _, p in candidates) + backoff_mass
            candidates = [(w, p / norm) for w, p in candidates]

        u = rng.random()
        acc = 0.0
        for w2, p in candidates:
            acc += p
            if u < acc:
                return w2
        # Хвост распределения — откат на континуационное распределение KN.
        # ``</s>`` из отката исключается: иначе генерация обрывается на первом
        # же шаге, потому что END — частая континуация в любом корпусе.
        tail = [(w, p) for w, p in self.p_cont.items() if w not in (END, UNK)]
        if not tail:
            return END
        return rng.choices([w for w, _ in tail], weights=[p for _, p in tail], k=1)[0]

    def generate(
        self,
        max_len: int = 40,
        temperature: float = 0.8,
        seed: int = 0,
    ) -> str:
        """Генерирует текст из ``<s>`` до ``</s>`` или лимита длины."""
        rng = random.Random(seed)
        current = START
        words: list[str] = []
        for _ in range(max_len):
            current = self._sample_next(current, temperature, rng)
            if current == END:
                break
            words.append(current)
        return " ".join(words)

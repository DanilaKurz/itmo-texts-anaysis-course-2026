"""Морфология и синтаксис: POS-статистика, пары из дерева, анафора.

Модуль держит всё, что зависит от spaCy, чтобы лабы 2–4 не дублировали загрузку
модели и обход дерева зависимостей.
"""

from __future__ import annotations

from collections import Counter

import pymorphy3
import spacy
from spacy.language import Language

# Местоимения третьего лица, которые разбираются в правиле анафоры.
THIRD_PERSON = {"он", "она", "оно", "они", "это"}

# Соответствие падежей UD и граммем pymorphy3 (у него творительный — ablt).
CASE_TO_GRAMMEME = {
    "Nom": "nomn",
    "Gen": "gent",
    "Dat": "datv",
    "Acc": "accs",
    "Ins": "ablt",
    "Loc": "loct",
}

_INFLECTOR = pymorphy3.MorphAnalyzer()


def load_nlp(parser: bool = True) -> Language:
    """Загружает русскую модель, при необходимости без парсера и NER.

    POS-статистике парсер не нужен, а его отключение экономит около трети
    времени на 9 млн токенов.
    """
    disable = [] if parser else ["parser", "ner"]
    return spacy.load("ru_core_news_sm", disable=disable)


# Служебные токены очистки: spaCy режет ``[NUM]`` на ``[`` + ``NUM`` + ``]``,
# поэтому фильтровать надо и по скобкам, и по «голому» верхнему регистру.
_SPECIAL_TOKENS = {"NUM", "URL", "EMAIL", "HANDLE", "HASHTAG"}

# Временные существительные — частые ложные антецеденты: «он» в тексте про
# ноябрь не указывает на ноябрь, но род совпадает (мужской), и наивное правило
# цепляет ближайшего кандидата.
_TEMPORAL_NOUNS = {"год", "время", "раз", "момент", "число", "месяц", "день", "этап"}


def _is_special(token) -> bool:
    """Служебные токены очистки (``[NUM]``, ``[URL]``) не участвуют в анализе.

    spaCy размечает их как NOUN/PROPN, и без фильтра они становятся
    «существительными» — в том числе антецедентами анафоры.
    """
    if token.text.startswith("["):
        return True
    return token.text.upper() in _SPECIAL_TOKENS and token.text.isupper()


def count_pos(doc) -> tuple[Counter, Counter]:
    """Считает POS-теги документа: UPOS и уточнённые теги.

    Returns:
        tuple[Counter, Counter]: ``(по токенам, по документам)``. Второй
        счётчик нужен, чтобы отделить «слово часто встречается» от «слово
        встречается во многих документах».
    """
    tokens: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    seen: set[str] = set()
    for token in doc:
        if token.is_space or token.is_punct or _is_special(token):
            continue
        tokens[token.pos_] += 1
        seen.add(token.pos_)
    docs.update(seen)
    return tokens, docs


# --------------------------------------------------------------- пары из дерева
def extract_pairs(doc) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Извлекает смысловые пары прямо из дерева зависимостей.

    Для новостей это две конструкции:

    * **субъект + действие** — связь ``nsubj`` от существительного к глаголу:
      «акции упали», «банк заблокировал»;
    * **аспект + оценка** — связь ``amod`` от прилагательного к
      существительному: «экран яркий», «персонал грубый».

    Местоимения в роли субъекта пропускаются: «он сообщил» не несёт смысла без
    разрешения анафоры, а оно делается отдельным шагом.
    """
    pairs_sv: list[tuple[str, str]] = []
    pairs_an: list[tuple[str, str]] = []
    for token in doc:
        if _is_special(token) or _is_special(token.head):
            continue
        if token.dep_ in ("nsubj", "nsubj:pass") and token.head.pos_ in ("VERB", "AUX"):
            if token.pos_ in ("NOUN", "PROPN"):
                pairs_sv.append((token.lemma_.lower(), token.head.lemma_.lower()))
        elif token.dep_ == "amod" and token.head.pos_ == "NOUN":
            pairs_an.append((token.head.lemma_.lower(), token.lemma_.lower()))
    return pairs_sv, pairs_an


# -------------------------------------------------------------------- анафора
def _inflect(noun: str, case: str | None, original: str = "") -> str:
    """Ставит существительное в нужный падеж и сохраняет заглавную букву."""
    form = noun
    if case and case in CASE_TO_GRAMMEME:
        parsed = _INFLECTOR.parse(noun)[0]
        inflected = parsed.inflect({CASE_TO_GRAMMEME[case], parsed.tag.number or "sing"})
        if inflected:
            form = inflected.word
    if original[:1].isupper():
        form = form[:1].upper() + form[1:]
    return form


def resolve_anaphora(doc, max_distance: int = 300) -> list[dict]:
    """Правиловый разбор местоимений третьего лица.

    Правило простое и объяснимое:

    1. кандидат — существительное или имя собственное **до** местоимения в
       текущем или двух предыдущих предложениях;
    2. род и число должны совпадать с местоимением (если они у местоимения
       выражены);
    3. из подходящих выбирается ближайшее, с бонусом за позицию подлежащего и
       за то же предложение.

    Возвращает список разборов с готовой заменой в правильном падеже.
    """
    sentences = list(doc.sents)
    results: list[dict] = []

    for sent_index, sentence in enumerate(sentences):
        for pronoun in sentence:
            # DET — это притяжательная форма («его компания»), её нельзя
            # заменить существительным без перестройки фразы, поэтому такие
            # случаи только считаются, но не заменяются.
            if pronoun.pos_ != "PRON":
                continue
            if pronoun.lemma_.lower() not in THIRD_PERSON:
                continue

            gender = pronoun.morph.get("Gender")
            number = pronoun.morph.get("Number")
            case = pronoun.morph.get("Case")

            best = None
            for prev_index in range(sent_index, max(-1, sent_index - 3), -1):
                for candidate in sentences[prev_index]:
                    if candidate.pos_ not in ("NOUN", "PROPN"):
                        continue
                    if _is_special(candidate):
                        continue
                    if candidate.lemma_.lower() in _TEMPORAL_NOUNS:
                        continue
                    if candidate.i >= pronoun.i:
                        continue
                    if pronoun.i - candidate.i > max_distance:
                        continue

                    cand_gender = candidate.morph.get("Gender")
                    cand_number = candidate.morph.get("Number")
                    if number and cand_number and number != cand_number:
                        continue
                    if gender and cand_gender and gender != cand_gender:
                        continue

                    # Близость важнее роли: штраф за предложение назад заметно
                    # больше бонуса за подлежащее, иначе антецедент «перескакивает»
                    # через границу предложения к более «важному» субъекту.
                    score = pronoun.i - candidate.i
                    score += 200 * (sent_index - prev_index)
                    score -= 5 if candidate.dep_ in ("nsubj", "nsubj:pass") else 0
                    if best is None or score < best[0]:
                        best = (score, candidate, prev_index)

            if best is None:
                continue

            _, antecedent, prev_index = best
            results.append(
                {
                    "pronoun": pronoun.text,
                    "pronoun_lemma": pronoun.lemma_.lower(),
                    "antecedent": antecedent.text,
                    "antecedent_lemma": antecedent.lemma_,
                    "replacement": _inflect(antecedent.lemma_, case[0] if case else None, antecedent.text),
                    "distance": pronoun.i - antecedent.i,
                    "cross_sentence": prev_index < sent_index,
                    "sentence": sentence.text,
                    "sentence_index": sent_index,
                    "pronoun_index": pronoun.i,
                    # «это» чаще указывает на целое событие, а не на конкретное
                    # существительное, поэтому такие замены помечаются как
                    # неуверенные — их разбираем отдельно в отчёте.
                    "confident": pronoun.lemma_.lower() != "это",
                }
            )
    return results


def apply_resolution_to_span(span, resolutions: list[dict]) -> str:
    """Собирает текст фрагмента с заменами (для показа «до/после»)."""
    mapping = {r["pronoun_index"]: r["replacement"] for r in resolutions}
    parts: list[str] = []
    for token in span:
        text = mapping.get(token.i, token.text)
        parts.append(text + token.whitespace_)
    return "".join(parts).strip()


def apply_resolution(doc, resolutions: list[dict]) -> str:
    """Собирает текст документа с заменами разрешённых местоимений."""
    mapping = {r["pronoun_index"]: r["replacement"] for r in resolutions}
    parts: list[str] = []
    for token in doc:
        text = mapping.get(token.i, token.text)
        parts.append(text + token.whitespace_)
    return "".join(parts).strip()

"""Обращение по имени и разбор «да/нет». Имя ищется в тексте распознавания (нечётко)."""
import re
from difflib import SequenceMatcher


def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е"))


def split_wake(text: str, words: list[str], ratio: float = 0.75) -> tuple[bool, str]:
    """Ищет имя в первых трёх словах или в последнем слове. Возвращает (найдено, текст без имени)."""
    tokens = text.split()
    if not tokens:
        return False, ""
    wake = [_norm(w) for w in words]
    candidates = list(dict.fromkeys(list(range(min(3, len(tokens)))) + [len(tokens) - 1]))
    for i in candidates:
        n = _norm(tokens[i])
        if n and any(n == w or SequenceMatcher(None, n, w).ratio() >= ratio for w in wake):
            rest = tokens[:i] + tokens[i + 1:]
            return True, " ".join(rest).strip(" ,.!?;:-")
    return False, text


_YES = {"да", "давай", "ага", "угу", "конечно", "подтверждаю", "хорошо", "ладно", "делай",
        "ок", "окей", "разрешаю", "валяй", "yes"}
_NO = {"нет", "не", "отмена", "стоп", "отставить", "неа", "no"}


def parse_yes_no(text: str) -> bool | None:
    tokens = {_norm(t) for t in text.split()}
    if tokens & _NO:
        return False
    if tokens & _YES:
        return True
    return None

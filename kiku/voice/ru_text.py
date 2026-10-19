"""Подготовка русского текста к синтезу: числа, время и даты словами.

Silero не читает цифры, поэтому всё, что в ответе записано цифрами («19 октября», «14:30», «3 задачи»), превращаем в слова.
"""
import re

_UNITS_M = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать",
            "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
_ORD_UNITS = ["", "первое", "второе", "третье", "четвёртое", "пятое", "шестое", "седьмое", "восьмое", "девятое", "десятое",
              "одиннадцатое", "двенадцатое", "тринадцатое", "четырнадцатое", "пятнадцатое", "шестнадцатое",
              "семнадцатое", "восемнадцатое", "девятнадцатое"]
_ORD_TENS = {20: "двадцатое", 30: "тридцатое"}
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
           "сентября", "октября", "ноября", "декабря"]


def _below_1000(n: int, feminine: bool) -> list[str]:
    words = [_HUNDREDS[n // 100]] if n >= 100 else []
    n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
    if n:
        words.append({1: "одна", 2: "две"}[n] if feminine and n in (1, 2) else _UNITS_M[n])
    return [w for w in words if w]


def plural(n: int, one: str, few: str, many: str) -> str:
    """1 час, 2 часа, 5 часов."""
    if 11 <= n % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def number_words(n: int, feminine: bool = False) -> str:
    """Целое число словами (до миллиарда). feminine: «одна», «две» — для минут, задач и т. п."""
    if n < 0:
        return "минус " + number_words(-n, feminine)
    if n == 0:
        return _UNITS_M[0]
    if n >= 10**9:
        return " ".join(_UNITS_M[int(d)] for d in str(n))
    words: list[str] = []
    for scale, forms, fem in ((10**6, ("миллион", "миллиона", "миллионов"), False),
                              (10**3, ("тысяча", "тысячи", "тысяч"), True)):
        part = n // scale % 1000
        if part:
            words += _below_1000(part, fem) + [plural(part, *forms)]
    words += _below_1000(n % 1000, feminine)
    return " ".join(words)


def day_words(day: int) -> str:
    """Число месяца как порядковое: «девятнадцатое», «двадцать первое»."""
    if day < 20:
        return _ORD_UNITS[day]
    tens, unit = day // 10 * 10, day % 10
    return _ORD_TENS[tens] if not unit else f"{_TENS[tens // 10]} {_ORD_UNITS[unit]}"


def _time_words(h: int, m: int) -> str:
    hours = f"{number_words(h)} {plural(h, 'час', 'часа', 'часов')}"
    if m == 0:
        return f"{hours} ровно"
    return f"{hours} {number_words(m, feminine=True)} {plural(m, 'минута', 'минуты', 'минут')}"


def speakable(text: str) -> str:
    """Заменяет цифры словами: время, даты («2026-10-20», «19 октября») и просто числа."""
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lambda m: _time_words(int(m[1]), int(m[2])), text)
    text = re.sub(r"\b\d{4}-(\d{2})-(\d{2})\b",
                  lambda m: f"{day_words(int(m[2]))} {MONTHS[int(m[1]) - 1]}" if 1 <= int(m[1]) <= 12 and 1 <= int(m[2]) <= 31
                  else m[0], text)
    months = "|".join(MONTHS)
    text = re.sub(rf"\b(\d{{1,2}})(\s+(?:{months}))\b",
                  lambda m: day_words(int(m[1])) + m[2] if 1 <= int(m[1]) <= 31 else m[0], text)
    return re.sub(r"\d+", lambda m: number_words(int(m[0])), text)

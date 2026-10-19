"""Разбор времени из русской речи: «через двадцать минут», «в 18:30», «завтра в девять утра», «2026-10-20 09:00».

Распознавание (Whisper) отдаёт то цифры, то слова — понимаем оба варианта. Без внешних библиотек.
"""
import re
from datetime import datetime, timedelta

from .voice.ru_text import MONTHS, plural

_UNITS = {w: i for i, w in enumerate(
    "ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать "
    "пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split())}
_UNITS.update({"одна": 1, "одну": 1, "две": 2})
_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
         "семьдесят": 70, "восемьдесят": 80, "девяносто": 90}
_DAYS = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
_DAYPART_HOUR = {"утром": 9, "днем": 13, "вечером": 19}


def tokenize(text: str) -> list[str]:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"(?<=\d)[:.](?=\d)", " ", t)          # 18:30 -> «18 30»
    return re.findall(r"[a-zа-я0-9\-]+", t)


def _number(tokens: list[str], i: int) -> tuple[int | None, int]:
    """Число цифрами или словами (до ста). Возвращает (значение, индекс после числа)."""
    if i >= len(tokens):
        return None, i
    if tokens[i].isdigit():
        return int(tokens[i]), i + 1
    total, j = None, i
    if tokens[j] == "сто":
        total, j = 100, j + 1
    if j < len(tokens) and tokens[j] in _TENS:
        total, j = (total or 0) + _TENS[tokens[j]], j + 1
    if j < len(tokens) and tokens[j] in _UNITS and (total is None or total % 10 == 0):
        total, j = (total or 0) + _UNITS[tokens[j]], j + 1
    return (total, j) if total is not None else (None, i)


def _unit_seconds(tok: str) -> int | None:
    if tok.startswith("сек"):
        return 1
    if tok.startswith("мин"):
        return 60
    if tok.startswith("час"):
        return 3600
    return None


def parse_duration(tokens: list[str], i: int = 0) -> tuple[int, int] | None:
    """«двадцать минут», «полтора часа», «1 час 30 минут», «полчаса». Возвращает (секунды, индекс после)."""
    total, found = 0.0, False
    while i < len(tokens):
        tok = tokens[i]
        if tok == "и" and found:
            i += 1
            continue
        if tok == "полчаса":
            total, i, found = total + 1800, i + 1, True
            continue
        if tok in ("полтора", "полторы") and i + 1 < len(tokens) and _unit_seconds(tokens[i + 1]):
            total, i, found = total + 1.5 * _unit_seconds(tokens[i + 1]), i + 2, True
            continue
        n, j = _number(tokens, i)
        if n is None:
            n, j = 1, i                                       # «через час»
        if j < len(tokens) and _unit_seconds(tokens[j]):
            total, i, found = total + n * _unit_seconds(tokens[j]), j + 1, True
        else:
            break
    return (int(total), i) if found else None


def _clock(tokens: list[str], i: int) -> tuple[int, int, int] | None:
    """«18 30», «шесть тридцать вечера», «9 утра». Возвращает (час, минута, индекс после)."""
    hour, j = _number(tokens, i)
    if hour is None or hour > 24:
        return None
    minute = 0
    m, k = _number(tokens, j)
    if m is not None and m < 60 and (tokens[j].isdigit() and len(tokens[j]) == 2 or m >= 10 or tokens[j] == "ноль"):
        minute, j = m, k
        if tokens[j - 1] == "ноль" and j < len(tokens):       # «ноль пять»
            m2, k2 = _number(tokens, j)
            if m2 is not None and m2 < 10:
                minute, j = m2, k2
    part = tokens[j] if j < len(tokens) else ""
    if part in ("утра", "ночи"):
        hour = 0 if hour == 12 else hour
        j += 1
    elif part in ("дня", "вечера"):
        hour = hour + 12 if hour < 12 else hour
        j += 1
    hour %= 24
    return hour, minute, j


def _scan(tokens: list[str], now: datetime, allow_bare: bool) -> tuple[datetime, int, int] | None:
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok):
            try:
                day = datetime.strptime(tok, "%Y-%m-%d")
            except ValueError:
                continue
            k = i + 1 + (1 if i + 1 < len(tokens) and tokens[i + 1] in ("в", "t") else 0)
            clock = _clock(tokens, k)
            hour, minute, end = clock if clock else (9, 0, i + 1)
            return day.replace(hour=hour, minute=minute), i, end
        if tok in _DAYS:
            day = (now + timedelta(days=_DAYS[tok])).replace(second=0, microsecond=0)
            k = i + 1 + (1 if i + 1 < len(tokens) and tokens[i + 1] in ("в", "к") else 0)
            clock = _clock(tokens, k)
            if clock:
                return day.replace(hour=clock[0], minute=clock[1]), i, clock[2]
            if i + 1 < len(tokens) and tokens[i + 1] in _DAYPART_HOUR:
                return day.replace(hour=_DAYPART_HOUR[tokens[i + 1]], minute=0), i, i + 2
            continue
        if tok == "через":
            dur = parse_duration(tokens, i + 1)
            if dur:
                return now + timedelta(seconds=dur[0]), i, dur[1]
            continue
        if tok in ("в", "к"):
            clock = _clock(tokens, i + 1)
            if clock:
                dt = now.replace(hour=clock[0], minute=clock[1], second=0, microsecond=0)
                return (dt if dt > now else dt + timedelta(days=1)), i, clock[2]
            continue
        if allow_bare:
            dur = parse_duration(tokens, i)
            if dur:
                return now + timedelta(seconds=dur[0]), i, dur[1]
    return None


def parse_when(text: str, now: datetime | None = None) -> datetime | None:
    """Момент времени из фразы (длительность тоже годится: «20 минут»). None — не поняли."""
    found = _scan(tokenize(text), now or datetime.now(), allow_bare=True)
    return found[0] if found else None


def extract_when(text: str, now: datetime | None = None) -> tuple[datetime, str, str] | None:
    """Находит выражение времени внутри фразы. Возвращает (момент, остаток фразы без него, само выражение времени)."""
    tokens = tokenize(text)
    found = _scan(tokens, now or datetime.now(), allow_bare=False)
    if not found:
        return None
    dt, start, end = found
    return dt, " ".join(tokens[:start] + tokens[end:]), " ".join(tokens[start:end])


def parse_seconds(text: str) -> int | None:
    """Длительность в секундах: «20 минут», «полчаса», «1 час 30 минут»."""
    tokens = tokenize(text)
    for i in range(len(tokens)):
        dur = parse_duration(tokens, i)
        if dur:
            return dur[0]
    return None


def describe_when(dt: datetime, now: datetime | None = None) -> str:
    """Для голоса: «через 20 минут», «сегодня в 18:30», «завтра в 9:00», «20 октября в 18:30»."""
    now = now or datetime.now()
    secs = int((dt - now).total_seconds())
    if secs < 120 * 60:
        mins = max(1, round(secs / 60))
        return f"через {mins} {plural(mins, 'минуту', 'минуты', 'минут')}"
    clock = f"{dt.hour}:{dt.minute:02d}"
    days = (dt.date() - now.date()).days
    if days == 0:
        return f"сегодня в {clock}"
    if days == 1:
        return f"завтра в {clock}"
    return f"{dt.day} {MONTHS[dt.month - 1]} в {clock}"


def describe_seconds(secs: int) -> str:
    """90 -> «1 минуту 30 секунд»; для озвучки таймера."""
    h, rest = divmod(int(secs), 3600)
    m, s = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} {plural(h, 'час', 'часа', 'часов')}")
    if m:
        parts.append(f"{m} {plural(m, 'минуту', 'минуты', 'минут')}")
    if s or not parts:
        parts.append(f"{s} {plural(s, 'секунду', 'секунды', 'секунд')}")
    return " ".join(parts)

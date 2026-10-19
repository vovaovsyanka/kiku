"""Интернет «по запросу»: погода и курс валют. По умолчанию ВЫКЛЮЧЕН (config.yaml → internet.enabled).

Кику остаётся офлайн-помощницей: сеть нужна только этим двум инструментам, ключей и регистрации не требуется
(погода — open-meteo.com, курсы — Банк России). Ответы сервисов — внешние данные, поэтому помечены недоверенными.
"""
import json
import urllib.parse
import urllib.request

from ..voice.ru_text import plural
from .registry import ToolContext, ToolError, ToolRegistry

TIMEOUT_SEC = 8
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
CBR_URL = "https://www.cbr-xml-daily.ru/daily_json.js"

WEATHER_CODES = {0: "ясно", 1: "в основном ясно", 2: "переменная облачность", 3: "пасмурно", 45: "туман", 48: "изморозь",
                 51: "лёгкая морось", 53: "морось", 55: "сильная морось", 56: "ледяная морось", 57: "ледяная морось",
                 61: "небольшой дождь", 63: "дождь", 65: "сильный дождь", 66: "ледяной дождь", 67: "ледяной дождь",
                 71: "небольшой снег", 73: "снег", 75: "сильный снег", 77: "снежные зёрна", 80: "небольшой ливень",
                 81: "ливень", 82: "сильный ливень", 85: "снегопад", 86: "сильный снегопад", 95: "гроза",
                 96: "гроза с градом", 99: "гроза с сильным градом"}
CURRENCY_ALIASES = {"доллар": "USD", "доллара": "USD", "долларов": "USD", "usd": "USD", "евро": "EUR", "eur": "EUR",
                    "юань": "CNY", "юаня": "CNY", "юаней": "CNY", "cny": "CNY", "фунт": "GBP", "фунта": "GBP", "gbp": "GBP"}
CURRENCY_NAMES = {"USD": "Доллар", "EUR": "Евро", "CNY": "Юань", "GBP": "Фунт"}


def temp_words(t: float) -> str:
    n = round(t)
    return f"плюс {n}" if n > 0 else f"минус {-n}" if n < 0 else "ноль"


def rubles_words(value: float) -> str:
    rub, kop = int(value), round((value - int(value)) * 100)
    parts = [f"{rub} {plural(rub, 'рубль', 'рубля', 'рублей')}"] if rub or not kop else []
    if kop:
        parts.append(f"{kop} {plural(kop, 'копейка', 'копейки', 'копеек')}")
    return " ".join(parts)


def city_forms(city: str) -> list[str]:
    """«в Москве» -> Москве, Москва; «в Казани» -> Казань; «в Петербурге» -> Петербург: геокодер ищет начальную форму."""
    c = city.strip()
    forms = []
    if len(c) > 3:
        stem = c[:-1]
        if c[-1] in "еу":
            forms += [stem + "а", stem + "я", stem]
        elif c[-1] == "и":
            forms += [stem + "ь", stem + "я", stem + "а"]
    return forms + [c]


def fetch_json(url: str, params: dict | None = None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "kiku-assistant"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:    # noqa: S310 — адреса зашиты выше
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"нет связи с сервисом ({type(e).__name__})") from e


def register_net_tools(reg: ToolRegistry, ctx: ToolContext, fetch=fetch_json) -> None:
    net = getattr(ctx.cfg, "internet", None)

    def need_internet():
        if not (net and getattr(net, "enabled", False)):
            raise ToolError("интернет выключен: чтобы узнавать погоду и курсы, поставьте internet.enabled: true в config.yaml")

    @reg.tool("weather", "Погода сейчас и на сегодня. city — город (если не назван, берётся город по умолчанию из настроек). "
                         "Работает, только если разрешён интернет.", reads=True, untrusted=True)
    def weather(city: str = "") -> dict:
        need_internet()
        city = (city or getattr(net, "city", "") or "").strip()
        if not city:
            raise ToolError("не знаю, для какого города: назовите город или впишите internet.city в config.yaml")
        place = None
        for form in city_forms(city):             # геокодер ищет «по началу слова» («Москве» -> «Москверо»), поэтому берём точное совпадение
            found = fetch(GEO_URL, {"name": form, "count": 5, "language": "ru", "format": "json"}).get("results") or []
            place = next((p for p in found if p.get("name", "").lower() == form.lower()), None)
            if place:
                break
        if place is None:
            raise ToolError(f"не нашла город «{city}»")
        data = fetch(WEATHER_URL, {
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 1,
            "wind_speed_unit": "ms",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
        cur, day = data["current"], data["daily"]
        cond = WEATHER_CODES.get(int(cur.get("weather_code", -1)), "")
        rain = day.get("precipitation_probability_max", [None])[0]
        name = place.get("name", city)
        summary = (f"Сейчас в городе {name} {temp_words(cur['temperature_2m'])}, {cond}. "
                   f"Днём до {temp_words(day['temperature_2m_max'][0])}, ночью {temp_words(day['temperature_2m_min'][0])}.")
        if rain is not None and rain >= 30:
            summary += f" Вероятность осадков {rain} {plural(rain, 'процент', 'процента', 'процентов')}."
        return {"city": name, "summary": summary}

    @reg.tool("currency", "Курс валюты к рублю по данным Банка России. code: USD | EUR | CNY | GBP (или «доллар», «евро»). "
                          "Работает, только если разрешён интернет.", reads=True, untrusted=True)
    def currency(code: str = "USD") -> dict:
        need_internet()
        key = CURRENCY_ALIASES.get(code.strip().lower(), code.strip().upper())
        rates = fetch(CBR_URL).get("Valute", {})
        if key not in rates:
            raise ToolError(f"курса «{code}» нет; доступны: {', '.join(sorted(rates)[:12])}…")
        r = rates[key]
        value = r["Value"] / r["Nominal"]
        name = CURRENCY_NAMES.get(key, key)
        summary = f"{name} стоит {rubles_words(value)}"
        diff = value - r["Previous"] / r["Nominal"]
        if abs(diff) >= 0.01:
            summary += f", {'вырос' if diff > 0 else 'упал'} за день на {rubles_words(abs(diff))}"
        return {"code": key, "rub": round(value, 2), "summary": summary + "."}

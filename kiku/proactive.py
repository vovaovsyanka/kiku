"""Инициатива Кику: утренняя сводка и забота о перерывах.

Сводка — «Доброе утро, хозяин! Сегодня… Открытых задач… Напоминаю…»: произносится при первом запуске за день
(proactive.morning_briefing) и по просьбе («сводка», «доброе утро»). Перерывы — если вы непрерывно за компьютером
дольше proactive.break_every_min минут, Кику мягко предлагает размяться (по простою мыши и клавиатуры в Windows).
"""
import logging
import random
import sys
import threading
import time
from datetime import datetime

from .persona import DAYS
from .timeparse import describe_seconds, describe_when
from .voice.ru_text import MONTHS

log = logging.getLogger("kiku.proactive")

IDLE_RESET_SEC = 300          # столько тишины на клавиатуре и мыши = хозяин отошёл, отсчёт активности начинается заново
CHECK_EVERY_SEC = 30

BREAK_PHRASES = [
    "{owner}, вы за компьютером уже {elapsed}. Может, разомнётесь и попьёте воды?",
    "Простите, что отвлекаю, {owner}: вы сидите уже {elapsed}. Самое время отдохнуть глазам.",
    "{owner}, уже {elapsed} без перерыва. Давайте сделаем небольшую паузу?",
]


def greeting(hour: int) -> str:
    if hour < 5:
        return "Доброй ночи"
    if hour < 12:
        return "Доброе утро"
    if hour < 18:
        return "Добрый день"
    return "Добрый вечер"


def briefing_text(cfg, memory, registry=None, now: datetime | None = None) -> str:
    """Сводка на сегодня из задач, напоминаний и (если разрешён интернет) погоды."""
    now = now or datetime.now()
    owner = cfg.persona.owner_title
    today = now.strftime("%Y-%m-%d")
    parts = [f"{greeting(now.hour)}, {owner}! Сегодня {DAYS[now.weekday()]}, {now.day} {MONTHS[now.month - 1]}."]

    tasks = memory.list_tasks("open")
    if not tasks:
        parts.append("Открытых задач нет.")
    else:
        overdue = [t for t in tasks if t["due"] and t["due"][:10] < today]
        due_today = [t for t in tasks if t["due"] and t["due"][:10] == today]
        parts.append(f"Открытых задач: {len(tasks)}.")
        if due_today:
            parts.append("На сегодня: " + "; ".join(t["title"] for t in due_today[:3]) + ".")
        if overdue:
            parts.append("Просрочено: " + "; ".join(t["title"] for t in overdue[:3]) + ".")

    day_end = now.replace(hour=23, minute=59, second=59).timestamp()
    reminders = [r for r in memory.list_reminders() if r["kind"] == "reminder" and r["due_ts"] <= day_end]
    if reminders:
        first = reminders[0]
        parts.append(f"Напомню сегодня: {first['text']}, {describe_when(datetime.fromtimestamp(first['due_ts']), now)}"
                     + (f", и ещё {len(reminders) - 1}." if len(reminders) > 1 else "."))

    net = getattr(cfg, "internet", None)
    if registry is not None and net and getattr(net, "enabled", False) and getattr(net, "city", ""):
        res = registry.call("weather", {})
        if res.get("ok"):
            parts.append(res["result"]["summary"])
    return " ".join(parts)


def first_briefing_today(memory, now: datetime | None = None) -> bool:
    """True, если сегодня сводка ещё не звучала (и отмечает, что прозвучит)."""
    today = (now or datetime.now()).strftime("%Y-%m-%d")
    if memory.kv_get("briefing_date") == today:
        return False
    memory.kv_set("briefing_date", today)
    return True


def seconds_since_input() -> float | None:
    """Сколько секунд хозяин не трогал мышь и клавиатуру. None — узнать нельзя (не Windows)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        return (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0
    except Exception:  # noqa: BLE001
        return None


class BreakReminder:
    """Следит за непрерывной работой и зовёт отдохнуть. tick() отдельно от потока — чтобы тестировать."""
    def __init__(self, announce, owner: str, every_min: int, quiet_hours=(23, 7), idle_fn=seconds_since_input):
        self.announce, self.owner, self.every_sec = announce, owner, int(every_min) * 60
        self.quiet = quiet_hours
        self.idle_fn = idle_fn
        self.active_since: float | None = None

    def _quiet_now(self, now: datetime) -> bool:
        start, end = self.quiet
        return now.hour >= start or now.hour < end if start > end else start <= now.hour < end

    def tick(self, now_ts: float | None = None, now: datetime | None = None) -> bool:
        if self.every_sec <= 0:
            return False
        now_ts = time.time() if now_ts is None else now_ts
        idle = self.idle_fn()
        if idle is None or idle >= IDLE_RESET_SEC:
            self.active_since = None
            return False
        if self.active_since is None:
            self.active_since = now_ts
            return False
        if now_ts - self.active_since < self.every_sec or self._quiet_now(now or datetime.fromtimestamp(now_ts)):
            return False
        elapsed = describe_seconds(int((now_ts - self.active_since) // 60 * 60))
        self.active_since = now_ts                      # следующее напоминание — через такой же срок
        self.announce(random.choice(BREAK_PHRASES).format(owner=self.owner.capitalize(), elapsed=elapsed))
        return True


class Proactive:
    """Фоновый поток: раз в полминуты проверяет, не пора ли предложить перерыв."""
    def __init__(self, break_reminder: BreakReminder):
        self.breaks = break_reminder
        self._stop = threading.Event()

    def start(self) -> None:
        if self.breaks.every_sec > 0 and self.breaks.idle_fn() is not None:
            threading.Thread(target=self._loop, daemon=True, name="kiku-proactive").start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(CHECK_EVERY_SEC):
            try:
                self.breaks.tick()
            except Exception:  # noqa: BLE001
                log.exception("Ошибка проактивной проверки")

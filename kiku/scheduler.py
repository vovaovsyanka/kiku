"""Планировщик: раз в секунду проверяет базу и озвучивает сработавшие напоминания и таймеры.

Напоминания лежат в SQLite (memory.reminders), поэтому переживают перезапуск: если срок прошёл, пока Кику была
выключена, она скажет об этом при следующем запуске. Сам планировщик ничего не говорит — он вызывает fire(text).
"""
import logging
import threading
import time
from datetime import datetime

log = logging.getLogger("kiku.scheduler")

LATE_AFTER_SEC = 120      # опоздание больше этого — в тексте упоминается, когда просили напомнить


def reminder_phrase(item: dict, now_ts: float, owner: str) -> str:
    """Что сказать вслух для сработавшего напоминания (item — строка из memory.due_reminders)."""
    if item["kind"] == "timer":
        return f"Таймер сработал{': ' + item['text'] if item['text'] else ''}."
    late = now_ts - item["due_ts"]
    if late > LATE_AFTER_SEC:
        when = datetime.fromtimestamp(item["due_ts"]).strftime("%H:%M")
        return f"{owner.capitalize()}, вы просили напомнить в {when}: {item['text']}."
    return f"{owner.capitalize()}, напоминаю: {item['text']}."


class Scheduler:
    def __init__(self, memory, fire, owner: str = "хозяин", interval: float = 1.0):
        self.memory, self.fire, self.owner, self.interval = memory, fire, owner, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="kiku-scheduler")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def tick(self, now_ts: float | None = None) -> int:
        """Одна проверка (отдельно от потока — чтобы тестировать). Возвращает число сработавших."""
        now_ts = time.time() if now_ts is None else now_ts
        due = self.memory.due_reminders(now_ts)
        for item in due:
            self.fire(reminder_phrase(item, now_ts, self.owner))
        return len(due)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — например, база закрыта при выходе
                log.exception("Ошибка планировщика")
                if self._stop.wait(5):
                    return

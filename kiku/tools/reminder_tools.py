"""Напоминания и таймеры: Кику сама заговорит в нужный момент (см. scheduler.py)."""
from datetime import datetime

from ..timeparse import describe_seconds, describe_when, parse_seconds, parse_when
from .registry import ToolContext, ToolError, ToolRegistry

MAX_DAYS = 366


def register_reminder_tools(reg: ToolRegistry, ctx: ToolContext) -> None:
    mem = ctx.memory

    @reg.tool("set_reminder",
              "Напомнить в нужное время. text — о чём напомнить; when — время так, как сказал хозяин: "
              "«через 20 минут», «в 18:30», «завтра в 9 утра», «2026-10-20 09:00».")
    def set_reminder(text: str, when: str) -> dict:
        if not (text or "").strip():
            raise ToolError("о чём напомнить?")
        now = datetime.now()
        due = parse_when(when, now)
        if due is None:
            raise ToolError(f"не поняла время «{when}»; скажите, например, «через 20 минут» или «в 18:30»")
        if due <= now:
            raise ToolError("это время уже прошло")
        if (due - now).days > MAX_DAYS:
            raise ToolError("слишком далеко: напоминания ставлю не дальше чем на год")
        if ctx.dry_run:
            return {"dry_run": True, "when": describe_when(due, now)}
        rid = mem.add_reminder(text, due.timestamp())
        return {"reminder_id": rid, "when": describe_when(due, now)}

    @reg.tool("set_timer", "Поставить таймер. duration — «5 минут», «полчаса», «1 час 30 минут»; label — название (необязательно).")
    def set_timer(duration: str, label: str = "") -> dict:
        secs = parse_seconds(duration)
        if not secs:
            raise ToolError(f"не поняла длительность «{duration}»; скажите, например, «5 минут»")
        if secs > 24 * 3600:
            raise ToolError("таймер ставлю не больше чем на сутки")
        if ctx.dry_run:
            return {"dry_run": True, "seconds": secs}
        rid = mem.add_reminder(label or "", datetime.now().timestamp() + secs, kind="timer")
        return {"timer_id": rid, "duration": describe_seconds(secs)}

    @reg.tool("list_reminders", "Список предстоящих напоминаний и таймеров.", reads=True)
    def list_reminders() -> list:
        now = datetime.now()
        return [{"id": r["id"], "kind": r["kind"], "text": r["text"],
                 "when": describe_when(datetime.fromtimestamp(r["due_ts"]), now)} for r in mem.list_reminders()]

    @reg.tool("cancel_reminder", "Отменить напоминание или таймер по id (из list_reminders). id=0 — отменить все.")
    def cancel_reminder(reminder_id: int) -> str:
        if ctx.dry_run:
            return "[dry-run] отменила бы напоминание"
        if int(reminder_id) == 0:
            items = mem.list_reminders()
            for r in items:
                mem.cancel_reminder(r["id"])
            return f"отменено: {len(items)}"
        if not mem.cancel_reminder(int(reminder_id)):
            raise ToolError("такого напоминания нет")
        return "отменено"

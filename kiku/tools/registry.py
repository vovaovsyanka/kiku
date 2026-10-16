"""Реестр инструментов, которые может вызывать Кику.

Инструмент — обычная Python-функция. Её сигнатура автоматически превращается
в строку каталога для системного промпта, аргументы от модели приводятся к нужным типам.
ВАЖНО: в файлах с инструментами не использовать `from __future__ import annotations`.
"""
import inspect
import logging
import types
import typing
from dataclasses import dataclass, field
from typing import Callable

from ..audit import AuditLog

log = logging.getLogger("kiku.tools")


class ToolError(Exception):
    """Ожидаемая ошибка инструмента (текст будет показан модели и хозяину)."""


@dataclass
class ToolContext:
    """Всё, что нужно инструментам. UI подменяет confirm на свою реализацию."""
    cfg: object
    memory: object
    policy: object            # PathPolicy
    audit: AuditLog
    dry_run: bool = False
    confirm: Callable[[str], bool] = field(default=lambda question: False)


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    reads: bool = False       # True: результат нужен модели, чтобы ответить (поиск, списки, время)
    untrusted: bool = False   # True: результат содержит чужой текст (файл) — после него менять систему нельзя


def _coerce(value, hint):
    origin = typing.get_origin(hint)
    if origin in (typing.Union, getattr(types, "UnionType", ())):
        options = [a for a in typing.get_args(hint) if a is not type(None)]
        if value is None:
            return None
        hint = options[0] if options else str
    if hint is int:
        return int(value)
    if hint is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "да", "yes", "y")
        return bool(value)
    if hint is str:
        return value if value is None else str(value)
    return value


class ToolRegistry:
    def __init__(self, audit: AuditLog):
        self.tools: dict[str, Tool] = {}
        self.audit = audit

    def tool(self, name: str, description: str, reads: bool = False, untrusted: bool = False):
        def deco(fn):
            self.tools[name] = Tool(name, description, fn, reads, untrusted)
            return fn
        return deco

    def is_untrusted(self, name: str) -> bool:
        t = self.tools.get(name)
        return bool(t and t.untrusted)

    def is_read(self, name: str) -> bool:
        t = self.tools.get(name)
        return bool(t and t.reads)

    def names(self) -> list[str]:
        return list(self.tools)

    def catalog(self) -> str:
        lines = []
        for t in self.tools.values():
            sig = inspect.signature(t.fn)
            hints = typing.get_type_hints(t.fn)
            parts = []
            for p in sig.parameters.values():
                h = hints.get(p.name, str)
                hname = str(h).replace("typing.", "").replace("<class '", "").replace("'>", "")
                s = f"{p.name}: {hname}"
                if p.default is not inspect.Parameter.empty:
                    s += f" = {p.default!r}"
                parts.append(s)
            lines.append(f"- {t.name}({', '.join(parts)}) — {t.description}")
        return "\n".join(lines)

    def call(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"неизвестный инструмент «{name}»"}
        args = args if isinstance(args, dict) else {}
        try:
            hints = typing.get_type_hints(tool.fn)
            clean = {k: _coerce(v, hints.get(k, str)) for k, v in args.items()}
            inspect.signature(tool.fn).bind(**clean)
        except (TypeError, ValueError) as e:
            self.audit.write(tool=name, args=args, ok=False, error=f"bad args: {e}")
            return {"ok": False, "error": f"неверные аргументы: {e}"}
        try:
            result = tool.fn(**clean)
            out = {"ok": True, "result": result}
        except ToolError as e:
            out = {"ok": False, "error": str(e)}
        except PermissionError as e:
            out = {"ok": False, "error": f"нет доступа: {e}"}
        except Exception as e:  # noqa: BLE001 — модель должна увидеть ошибку, а не упасть
            log.exception("tool %s failed", name)
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.audit.write(tool=name, args=clean, **{k: v for k, v in out.items()})
        return out

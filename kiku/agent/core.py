"""Цикл агента: реплика хозяина -> один ответ модели с несколькими действиями -> (при необходимости) результаты -> итог.

Обычная команда («открой X и создай Y») обходится ОДНИМ вызовом модели. Второй нужен только если
действию требуется результат (поиск, список, время) или действие закончилось ошибкой.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..persona import build_system_prompt, now_ru

log = logging.getLogger("kiku.agent")

_NULLISH = {"", "null", "none", "нет", "n/a"}
MAX_ACTIONS = 4


@dataclass
class Call:
    tool: str
    args: dict


@dataclass
class Action:
    calls: list = field(default_factory=list)
    say: str = ""


def _loads_lenient(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _clean_tool(tool) -> Optional[str]:
    if isinstance(tool, str):
        tool = tool.strip()
        return None if tool.lower() in _NULLISH else tool
    return None


def parse_action(text: str) -> Action:
    data = _loads_lenient(text)
    if not isinstance(data, dict):
        return Action([], (text or "").strip())
    calls: list[Call] = []
    raw_actions = data.get("actions")
    if isinstance(raw_actions, list):
        for c in raw_actions[:MAX_ACTIONS]:
            if isinstance(c, dict):
                tool = _clean_tool(c.get("tool"))
                if tool:
                    calls.append(Call(tool, c["args"] if isinstance(c.get("args"), dict) else {}))
    else:  # старый формат с одним инструментом — на всякий случай
        tool = _clean_tool(data.get("tool"))
        if tool:
            calls.append(Call(tool, data["args"] if isinstance(data.get("args"), dict) else {}))
    say = data.get("say")
    return Action(calls, say.strip() if isinstance(say, str) else "")


class Agent:
    def __init__(self, llm, registry, memory, cfg, router=None):
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.cfg = cfg
        self.router = router
        self.history: list[dict] = []

    def _schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "maxItems": MAX_ACTIONS,
                    "items": {
                        "type": "object",
                        "properties": {"tool": {"enum": self.registry.names()}, "args": {"type": "object"}},
                        "required": ["tool", "args"],
                    },
                },
                "say": {"type": "string"},
            },
            "required": ["actions", "say"],
        }

    def _system(self) -> dict:
        facts = self.memory.list_facts(int(self.cfg.memory.max_facts_in_prompt))
        return {"role": "system",
                "content": build_system_prompt(self.cfg, self.registry.catalog(), facts)}

    def reset(self) -> None:
        self.history.clear()

    def warmup(self) -> None:
        """Заранее обработать системный промпт, чтобы первая команда не ждала разбора ~2000 токенов."""
        fn = getattr(self.llm, "warmup", None)
        if fn:
            fn([self._system(), {"role": "user", "content": f"[{now_ru()}] привет"}])

    def _remember_turn(self, user_msg: dict, final: str, raw: Optional[str] = None) -> None:
        """В историю кладём РОВНО те сообщения, которые видела модель (с меткой времени и её сырым JSON):
        тогда llama.cpp находит общее начало с прошлым запросом и не пересчитывает всю историю заново."""
        assistant = raw if raw else json.dumps({"actions": [], "say": final}, ensure_ascii=False)
        self.history += [user_msg, {"role": "assistant", "content": assistant}]
        self.history = self.history[-2 * int(self.cfg.llm.history_turns):]

    def respond(self, user_text: str, on_say: Optional[Callable[[str], None]] = None,
                on_tool: Optional[Callable[[str, dict, dict], None]] = None) -> str:
        """on_say вызывается для каждой фразы, которую нужно озвучить/показать. Возвращает финальную фразу."""
        say_cb = on_say or (lambda s: None)
        tool_cb = on_tool or (lambda *a: None)
        # Время идёт в сообщение хозяина, а не в системный промпт (чтобы не ломать кэш промпта).
        user_msg = {"role": "user", "content": f"[{now_ru()}] {user_text}"}

        # Быстрый путь: типовые команды выполняются без языковой модели.
        if self.router is not None:
            routed = self.router.try_route(user_text)
            if routed is not None:
                for tool, args, result in routed.steps:
                    tool_cb(tool, args, result)
                say_cb(routed.say)
                self._remember_turn(user_msg, routed.say)
                return routed.say

        system = self._system()
        turn = [user_msg]
        last_raw = None
        seen: dict = {}
        tainted = False   # в контексте есть текст из файла (недоверенный): менять систему больше нельзя
        final = ""

        for _ in range(int(self.cfg.llm.max_steps)):
            raw = self.llm.chat_json([system] + self.history + turn, schema=self._schema())
            last_raw = raw
            act = parse_action(raw)
            log.debug("LLM: %s", raw)

            if not act.calls:
                final = act.say or "Слушаю вас."
                say_cb(final)
                break

            if act.say:
                say_cb(act.say)   # озвучка начинается, пока выполняются действия

            results, need_more, looped = [], False, False
            for call in act.calls:
                key = (call.tool, json.dumps(call.args, sort_keys=True, ensure_ascii=False))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 2:         # один повтор допустим (например, поиск после индексации), третий — цикл
                    looped = True
                    break
                if tainted and not self.registry.allowed_when_tainted(call.tool):
                    res = {"ok": False, "error": "заблокировано: после чтения чужого текста (файл, буфер обмена) менять систему нельзя — "
                                                 "можно только отвечать хозяину или читать дальше"}
                else:
                    res = self.registry.call(call.tool, call.args)
                tool_cb(call.tool, call.args, res)
                results.append({"tool": call.tool, "result": res})
                if not res.get("ok"):
                    need_more = True      # модель должна увидеть ошибку и объяснить
                    break
                if self.registry.is_untrusted(call.tool):
                    tainted = True
                if self.registry.is_read(call.tool):
                    need_more = True      # модели нужен результат, чтобы ответить
            if looped:
                final = "Кажется, я хожу по кругу. Давайте попробуем сформулировать иначе."
                say_cb(final)
                break
            if not need_more:
                final = act.say or "Готово."
                if not act.say:
                    say_cb(final)
                break

            payload = json.dumps(results, ensure_ascii=False)
            if len(payload) > 3000:
                payload = payload[:3000] + "…(обрезано)"
            turn.append({"role": "assistant", "content": raw})
            turn.append({"role": "user", "content":
                         f"РЕЗУЛЬТАТЫ: {payload}\nОтветь итогом (при необходимости — ещё действия)."})
        else:
            final = "Не получилось довести дело до конца за разумное число шагов."
            say_cb(final)

        self._remember_turn(user_msg, final, last_raw if self._raw_matches(last_raw, final) else None)
        return final

    @staticmethod
    def _raw_matches(raw: Optional[str], final: str) -> bool:
        """Сырой ответ модели годится для истории, только если его say и есть итоговая фраза."""
        return bool(raw) and parse_action(raw).say.strip() == final.strip()

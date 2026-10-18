"""Быстрый путь: типовые команды выполняются шаблонами БЕЗ языковой модели (мгновенно и предсказуемо).

Всё, что не распознано, уходит к языковой модели. Включается/выключается в config.yaml → router.enabled.
Роутер вызывает те же инструменты с теми же проверками безопасности.
"""
import difflib
import re
from dataclasses import dataclass, field

from ..tools.paths import is_folder_alias

_FILLER = re.compile(r"^(?:(?:пожалуйста|кику|слушай|ну|а|и|давай|теперь|сейчас|еще|можешь|можешь ли ты|будь добра|будь добр)\s+)+")
_TAIL = re.compile(r"\s+(?:пожалуйста|кику)$")


def normalize(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _FILLER.sub("", t)
    t = _TAIL.sub("", t)
    return t.strip()


@dataclass
class Routed:
    say: str
    steps: list = field(default_factory=list)   # (tool, args, result)


_VOL_UP = {"громче", "погромче", "прибавь звук", "увеличь громкость", "сделай громче", "сделай погромче", "прибавь громкость"}
_VOL_DOWN = {"тише", "потише", "убавь звук", "уменьши громкость", "сделай тише", "сделай потише", "убавь громкость"}
_MUTE = {"без звука", "выключи звук", "отключи звук", "заглуши", "включи звук", "верни звук", "убери звук"}
_PAUSE = {"пауза", "поставь на паузу", "поставь музыку на паузу", "продолжи", "продолжи воспроизведение", "продолжай"}
_NEXT = {"следующий трек", "следующая песня", "следующий", "включи следующий трек", "переключи трек"}
_PREV = {"предыдущий трек", "предыдущая песня", "предыдущий", "включи предыдущий трек"}
_TIME = {"который час", "сколько времени", "сколько сейчас времени", "какое сегодня число", "какая сегодня дата",
         "какой сегодня день", "какой сегодня день недели"}
_TASKS = {"какие у меня задачи", "мои задачи", "что у меня по задачам", "список задач", "что мне нужно сделать",
          "какие задачи", "покажи задачи"}
_SHOW_DESKTOP = {"покажи рабочий стол", "сверни все окна", "сверни все", "сверни всё", "сверни все приложения"}

_OPEN = re.compile(r"^(?:открой|открыть|запусти|запустить|включи|стартани)\s+(?:папку\s+|приложение\s+|программу\s+|игру\s+)?(.+)$")
_MIN = re.compile(r"^(?:сверни|свернуть|минимизируй)\s+(?:окно\s+|приложение\s+|программу\s+)?(.+)$")
_MAX = re.compile(r"^(?:разверни|развернуть|максимизируй)\s+(?:окно\s+|приложение\s+|программу\s+)?(.+)$")
_FOCUS = re.compile(r"^(?:переключись на|перейди в|перейди на|покажи окно|вернись в|вернись к)\s+(.+)$")


class Router:
    def __init__(self, registry, cfg):
        self.registry = registry
        self.apps = {}
        for a in (getattr(cfg, "apps", None) or []):
            for key in [a.id, a.title, *getattr(a, "aliases", [])]:
                self.apps[normalize(str(key))] = a.title

    # ---------- вспомогательное ----------
    def _call(self, steps, tool, **args):
        res = self.registry.call(tool, args)
        steps.append((tool, args, res))
        return res

    def _find_app(self, target: str):
        if target in self.apps:
            return self.apps[target]
        close = difflib.get_close_matches(target, list(self.apps), n=1, cutoff=0.88)
        return self.apps[close[0]] if close else None

    @staticmethod
    def _err(res) -> str:
        return f"Не получилось: {res.get('error', 'неизвестная ошибка')}."

    # ---------- основной вход ----------
    def try_route(self, text: str):
        t = normalize(text)
        if not t:
            return None
        steps: list = []

        if t in _VOL_UP or t in _VOL_DOWN or t in _MUTE:
            action = "up" if t in _VOL_UP else "down" if t in _VOL_DOWN else "mute"
            res = self._call(steps, "volume", action=action)
            say = {"up": "Делаю громче.", "down": "Делаю тише.", "mute": "Переключаю звук."}[action]
            return Routed(say if res["ok"] else self._err(res), steps)

        for words, action, say in ((_PAUSE, "play_pause", "Хорошо."), (_NEXT, "next", "Следующий трек."),
                                   (_PREV, "previous", "Предыдущий трек.")):
            if t in words:
                res = self._call(steps, "media", action=action)
                return Routed(say if res["ok"] else self._err(res), steps)

        if t in _SHOW_DESKTOP:
            res = self._call(steps, "minimize_all")
            return Routed("Сворачиваю все окна." if res["ok"] else self._err(res), steps)

        if t in _TIME:
            res = self._call(steps, "get_time")
            when = re.sub(r"\s\d{4}(?=,)", "", res["result"]) if res["ok"] else ""   # год в ответе лишний
            return Routed(f"Сейчас {when}." if res["ok"] else self._err(res), steps)

        if t in _TASKS:
            res = self._call(steps, "list_tasks", status="open")
            if not res["ok"]:
                return Routed(self._err(res), steps)
            tasks = res["result"]
            if not tasks:
                return Routed("Открытых задач нет.", steps)
            items = "; ".join(x["title"] + (f", срок {x['due']}" if x["due"] else "") for x in tasks[:5])
            more = f" И ещё {len(tasks) - 5}." if len(tasks) > 5 else ""
            return Routed(f"Открытых задач: {len(tasks)}. {items}.{more}", steps)

        m = _OPEN.match(t)
        if m:
            target = m.group(1).strip()
            app_title = self._find_app(target)
            if app_title:
                res = self._call(steps, "launch_app", name=target)
                return Routed(f"Запускаю {app_title}." if res["ok"] else self._err(res), steps)
            if is_folder_alias(target):
                res = self._call(steps, "open_folder", path=target)
                return Routed(f"Открываю {target}." if res["ok"] else self._err(res), steps)
            return None

        for rx, action, say in ((_MIN, "minimize", "Сворачиваю"), (_MAX, "maximize", "Разворачиваю"),
                                (_FOCUS, "focus", "Переключаюсь на")):
            m = rx.match(t)
            if m:
                target = m.group(1).strip()
                if action == "minimize" and target in ("все", "все окна", "всё"):
                    res = self._call(steps, "minimize_all")
                    return Routed("Сворачиваю все окна." if res["ok"] else self._err(res), steps)
                res = self._call(steps, "window_action", action=action, title=target)
                if not res["ok"] and str(res.get("error", "")).startswith("окно не найдено"):
                    return None   # пусть разбирается языковая модель
                return Routed(f"{say} {target}." if res["ok"] else self._err(res), steps)
        return None

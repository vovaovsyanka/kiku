"""Быстрый путь: типовые команды выполняются шаблонами БЕЗ языковой модели (мгновенно и предсказуемо).

Всё, что не распознано, уходит к языковой модели. Включается/выключается в config.yaml → router.enabled.
Роутер вызывает те же инструменты с теми же проверками безопасности.
"""
import difflib
import random
import re
from dataclasses import dataclass, field

from ..timeparse import extract_when
from ..tools.paths import is_folder_alias
from ..tools.pc_tools import describe_battery, describe_disks, describe_load
from ..tools.scene_tools import find_scene

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
_BRIEFING = {"сводка", "дай сводку", "мою сводку", "что на сегодня", "что у меня сегодня", "план на день", "план на сегодня",
             "доброе утро", "добрый день", "добрый вечер", "расскажи что на сегодня"}
_LOAD = {"что тормозит", "почему тормозит компьютер", "почему тормозит", "почему лагает", "почему лагает компьютер",
         "что грузит компьютер", "что грузит процессор", "что жрет память", "что ест память", "как там компьютер",
         "как компьютер", "нагрузка", "загрузка компьютера", "что так грузит", "почему все тормозит",
         "сколько памяти", "сколько оперативки", "сколько свободно памяти", "сколько оперативной памяти", "сколько озу"}
_DISK = {"сколько места на диске", "место на диске", "свободное место", "сколько свободного места", "сколько места осталось"}
_BATTERY = {"заряд батареи", "сколько заряда", "сколько батареи", "батарея", "сколько процентов заряда", "сколько осталось заряда"}
_CANCEL_REMINDERS = {"отмени все напоминания", "отмени напоминания", "отмени таймер", "отмени все таймеры",
                     "выключи таймер", "останови таймер", "удали все напоминания"}
_LIST_REMINDERS = {"какие у меня напоминания", "мои напоминания", "какие напоминания", "покажи напоминания", "какие таймеры",
                   "что ты мне напомнишь", "сколько осталось до таймера"}
_UNDO = {"верни как было", "отмени раскладку", "верни файлы обратно", "отмени уборку", "верни обратно"}

_DUPLICATES = re.compile(r"^(?:найди|покажи|есть ли)\s+(?:дубликаты|копии)(?:\s+(?:в|на)\s+(.+))?$")
_MOVE_DUPLICATES = re.compile(r"^(?:убери|убрать|переложи|собери)\s+(?:дубликаты|копии)(?:\s+(?:в|на)\s+(.+))?$")
_ORGANIZE = re.compile(r"^(?:разбери|разложи|наведи порядок|упорядочь|прибери|разберись с)\s+(?:файлы\s+|все\s+)?(?:(?:в|на)\s+)?(.+)$")
_OPEN = re.compile(r"^(?:открой|открыть|запусти|запустить|включи|стартани)\s+(?:папку\s+|приложение\s+|программу\s+|игру\s+)?(.+)$")
_MIN = re.compile(r"^(?:сверни|свернуть|минимизируй)\s+(?:окно\s+|приложение\s+|программу\s+)?(.+)$")
_MAX = re.compile(r"^(?:разверни|развернуть|максимизируй)\s+(?:окно\s+|приложение\s+|программу\s+)?(.+)$")
_FOCUS = re.compile(r"^(?:переключись на|перейди в|перейди на|покажи окно|вернись в|вернись к)\s+(.+)$")
_REMIND = re.compile(r"^(?:напомни|напомнить|поставь напоминание)\s+(?:мне\s+)?(.+)$")
_TIMER = re.compile(r"^(?:поставь\s+|засеки\s+|запусти\s+|включи\s+|установи\s+)?таймер\s+(?:на\s+)?(.+)$")
_WEATHER = re.compile(r"^(?:какая\s+)?погода(?:\s+(?:в|во|на)\s+(.+?))?(?:\s+(?:сегодня|сейчас))?$")
_CURRENCY = re.compile(r"^(?:какой\s+)?курс\s+(доллара|долларов|доллар|евро|юаня|юань|фунта|фунт)(?:\s+сегодня)?$")
_REMIND_LEAD = re.compile(r"^(?:о том что|про то что|о том чтобы|что|чтобы|о|про)\s+")

# Диктовка и дневник читают исходный текст (регистр и знаки препинания важны), а не нормализованный.
# Слово «напиши» намеренно не используется: «напиши стих» — это просьба к языковой модели, а не диктовка.
_SEP = r"[\s:,.—–-]+"
_DICTATE = re.compile(rf"^\W*(?:кику\W+)?(?:набери|напечатай|надиктуй|диктую|продиктую|введи)(?:\s+текст)?{_SEP}(.+)$", re.I | re.S)
_DIARY = re.compile(rf"^\W*(?:кику\W+)?(?:запиши|добавь|занеси)\s+в\s+дневник{_SEP}(.+)$", re.I | re.S)

_SMALL_TALK = [
    ({"привет", "здравствуй", "здравствуйте", "приветик", "хай"}, ["Здравствуйте, {owner}! Чем могу помочь?", "Привет, {owner}! Я слушаю."]),
    ({"спасибо", "благодарю", "спасибо большое", "большое спасибо", "спасибки"}, ["Всегда пожалуйста, {owner}!", "Рада помочь!", "Не за что, {owner}."]),
    ({"как дела", "как ты", "как жизнь", "как поживаешь", "как настроение"}, ["Всё хорошо, {owner}, жду поручений.", "Отлично, {owner}! А у вас как?"]),
    ({"ты тут", "ты здесь", "ты меня слышишь", "ты слышишь"}, ["Конечно, {owner}, я на месте!", "Да-да, слушаю вас, {owner}."]),
    ({"пока", "до свидания", "до встречи", "увидимся"}, ["До встречи, {owner}!", "Буду ждать, {owner}."]),
    ({"молодец", "умница", "ты молодец", "ты умница", "хорошая девочка"}, ["Спасибо, {owner}, мне очень приятно!", "Стараюсь, {owner}!"]),
]
_TALK = {phrase: replies for phrases, replies in _SMALL_TALK for phrase in phrases}


class Router:
    def __init__(self, registry, cfg, briefing_fn=None):
        self.registry = registry
        self.cfg = cfg
        self.briefing_fn = briefing_fn            # () -> текст утренней сводки
        self.owner = cfg.persona.owner_title
        self.apps = {}
        for a in (getattr(cfg, "apps", None) or []):
            for key in [a.id, a.title, *getattr(a, "aliases", [])]:
                self.apps[normalize(str(key))] = a.title

    # ---------- вспомогательное ----------
    def _call(self, steps, tool, /, **args):
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

    def _done(self, res, say: str, steps) -> Routed:
        return Routed(say if res["ok"] else self._err(res), steps)

    # ---------- основной вход ----------
    def try_route(self, text: str):
        t = normalize(text)
        if not t:
            return None
        steps: list = []
        for rule in (self._rule_raw_text, self._rule_system, self._rule_reminders, self._rule_scenes, self._rule_briefing,
                     self._rule_pc, self._rule_files, self._rule_net, self._rule_open, self._rule_windows, self._rule_talk):
            routed = rule(text, t, steps)
            if routed is not None:
                return routed
        return None

    # ---------- правила: каждое возвращает Routed или None ----------
    def _rule_raw_text(self, text, t, steps):
        m = _DICTATE.match(text.strip())
        if m:
            return self._done(self._call(steps, "type_text", text=m.group(1).strip()), "Печатаю.", steps)
        m = _DIARY.match(text.strip())
        if m:
            return self._done(self._call(steps, "diary_add", text=m.group(1).strip()), "Записала в дневник.", steps)
        return None

    def _rule_system(self, text, t, steps):
        if t in _VOL_UP or t in _VOL_DOWN or t in _MUTE:
            action = "up" if t in _VOL_UP else "down" if t in _VOL_DOWN else "mute"
            say = {"up": "Делаю громче.", "down": "Делаю тише.", "mute": "Переключаю звук."}[action]
            return self._done(self._call(steps, "volume", action=action), say, steps)

        for words, action, say in ((_PAUSE, "play_pause", "Хорошо."), (_NEXT, "next", "Следующий трек."),
                                   (_PREV, "previous", "Предыдущий трек.")):
            if t in words:
                return self._done(self._call(steps, "media", action=action), say, steps)

        if t in _SHOW_DESKTOP:
            return self._done(self._call(steps, "minimize_all"), "Сворачиваю все окна.", steps)

        if t in _TIME:
            res = self._call(steps, "get_time")
            when = re.sub(r"\s\d{4}(?=,)", "", res["result"]) if res["ok"] else ""      # год в ответе лишний
            return self._done(res, f"Сейчас {when}.", steps)

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
        return None

    def _rule_reminders(self, text, t, steps):
        if t in _CANCEL_REMINDERS:
            return self._done(self._call(steps, "cancel_reminder", reminder_id=0), "Отменила.", steps)
        if t in _LIST_REMINDERS:
            res = self._call(steps, "list_reminders")
            if not res["ok"]:
                return Routed(self._err(res), steps)
            items = res["result"]
            if not items:
                return Routed("Напоминаний и таймеров нет.", steps)
            more = f" Всего {len(items)}." if len(items) > 1 else ""
            return Routed(f"Ближайшее: {items[0]['text'] or 'таймер'}, {items[0]['when']}.{more}", steps)
        m = _TIMER.match(t)
        if m:
            res = self._call(steps, "set_timer", duration=m.group(1))
            return self._done(res, f"Таймер на {res['result']['duration']}." if res["ok"] else "", steps)
        m = _REMIND.match(t)
        if m:
            found = extract_when(m.group(1))
            if not found:
                return None                                      # времени не слышно — пусть разбирается модель
            _, what, when = found
            what = _REMIND_LEAD.sub("", what).strip()
            if not what:
                return Routed("О чём напомнить?", steps)
            res = self._call(steps, "set_reminder", text=what, when=when)
            return self._done(res, f"Хорошо, напомню {res['result']['when']}." if res["ok"] else "", steps)
        return None

    def _rule_scenes(self, text, t, steps):
        scene = find_scene(self.cfg, t) or find_scene(self.cfg, re.sub(r"^(?:включи|запусти|давай|активируй)\s+", "", t))
        if scene is None:
            return None
        errors = []
        for step in scene.steps:
            res = self._call(steps, step.tool, **(vars(step.args) if getattr(step, "args", None) is not None else {}))
            if not res["ok"]:
                errors.append(res.get("error"))
        if errors:
            return Routed(f"Режим «{scene.title}» включён не полностью: {errors[0]}.", steps)
        return Routed(getattr(scene, "say", None) or f"Включаю режим «{scene.title}».", steps)

    def _rule_briefing(self, text, t, steps):
        if t in _BRIEFING and self.briefing_fn is not None:
            return Routed(self.briefing_fn(), steps)
        return None

    def _rule_pc(self, text, t, steps):
        if t in _LOAD:
            status = self._call(steps, "system_status")
            if not status["ok"]:
                return Routed(self._err(status), steps)
            by_mem = self._call(steps, "top_processes", by="memory", limit=3)
            by_cpu = self._call(steps, "top_processes", by="cpu", limit=1)
            return Routed(describe_load(status["result"], by_mem.get("result", []), by_cpu.get("result", [])), steps)
        if t in _DISK or t in _BATTERY:
            status = self._call(steps, "system_status")
            if not status["ok"]:
                return Routed(self._err(status), steps)
            return Routed(describe_disks(status["result"]) if t in _DISK else describe_battery(status["result"]), steps)
        return None

    def _rule_files(self, text, t, steps):
        if t in _UNDO:
            res = self._call(steps, "undo_organize")
            return self._done(res, f"Вернула файлов: {res['result']['restored']}." if res["ok"] else "", steps)
        m = _DUPLICATES.match(t)
        if m:
            res = self._call(steps, "find_duplicates", path=m.group(1) or "downloads")
            if not res["ok"]:
                return Routed(self._err(res), steps)
            r = res["result"]
            if not r["groups"]:
                return Routed("Дубликатов не нашла.", steps)
            return Routed(f"Нашла лишних копий: {r['extra_copies']}, они занимают около {r['wasted_mb']} мегабайт. "
                          "Скажите «убери дубликаты», и я переложу их в отдельную папку.", steps)
        m = _MOVE_DUPLICATES.match(t)
        if m:
            res = self._call(steps, "move_duplicates", path=m.group(1) or "downloads")
            return self._done(res, f"Переложила копий: {res['result']['moved']}." if res["ok"] else "", steps)
        m = _ORGANIZE.match(t)
        if m and is_folder_alias(m.group(1).strip()):
            res = self._call(steps, "organize_folder", path=m.group(1).strip())
            if not res["ok"]:
                return Routed(self._err(res), steps)
            r = res["result"]
            return Routed(f"Разложила файлов: {r['moved']}. Если что, скажите «верни как было»." if r["moved"]
                          else r.get("note", "Раскладывать нечего."), steps)
        return None

    def _rule_net(self, text, t, steps):
        m = _WEATHER.match(t)
        if m:
            res = self._call(steps, "weather", city=(m.group(1) or "").strip())
            return self._done(res, res["result"]["summary"] if res["ok"] else "", steps)
        m = _CURRENCY.match(t)
        if m:
            res = self._call(steps, "currency", code=m.group(1))
            return self._done(res, res["result"]["summary"] if res["ok"] else "", steps)
        return None

    def _rule_open(self, text, t, steps):
        m = _OPEN.match(t)
        if not m:
            return None
        target = m.group(1).strip()
        app_title = self._find_app(target)
        if app_title:
            return self._done(self._call(steps, "launch_app", name=target), f"Запускаю {app_title}.", steps)
        if is_folder_alias(target):
            return self._done(self._call(steps, "open_folder", path=target), f"Открываю {target}.", steps)
        return None

    def _rule_windows(self, text, t, steps):
        for rx, action, say in ((_MIN, "minimize", "Сворачиваю"), (_MAX, "maximize", "Разворачиваю"),
                                (_FOCUS, "focus", "Переключаюсь на")):
            m = rx.match(t)
            if m:
                target = m.group(1).strip()
                if action == "minimize" and target in ("все", "все окна", "всё"):
                    return self._done(self._call(steps, "minimize_all"), "Сворачиваю все окна.", steps)
                res = self._call(steps, "window_action", action=action, title=target)
                if not res["ok"] and str(res.get("error", "")).startswith("окно не найдено"):
                    return None   # пусть разбирается языковая модель
                return self._done(res, f"{say} {target}.", steps)
        return None

    def _rule_talk(self, text, t, steps):
        replies = _TALK.get(t)
        return Routed(random.choice(replies).format(owner=self.owner), steps) if replies else None

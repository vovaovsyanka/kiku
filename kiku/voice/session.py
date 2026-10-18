"""Голосовая сессия: микрофон -> VAD -> ASR -> имя «Кику» -> агент -> TTS.

Не зависит от интерфейса: консоль и окно подключаются через Hooks.
Полудуплекс: пока Кику говорит, микрофон закрыт (так она не слышит саму себя).
Команды можно вводить и текстом (submit) — они не требуют имени.
Текст ответа показывается в тот момент, когда он начинает звучать (без озвучки — сразу).
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .wake import parse_yes_no, split_wake

log = logging.getLogger("kiku.session")

IDLE, LISTENING, THINKING, SPEAKING = "idle", "listening", "thinking", "speaking"
MIN_PHRASE_SEC = 0.25   # короче — это щелчок или шум, а не слово


def _noop(*a, **k):
    return None


@dataclass
class Hooks:
    state: Callable[[str], None] = _noop                  # idle | listening | thinking | speaking
    heard: Callable[[str, bool], None] = _noop            # распознанный текст, обращались ли к Кику
    answer: Callable[[str], None] = _noop                 # текст ответа
    status: Callable[[str], None] = _noop                 # служебные сообщения (тайминги)
    tool: Callable[[str, dict, dict], None] = _noop       # вызов инструмента


class VoiceSession:
    def __init__(self, app, asr, listener, tts=None, hooks: Hooks | None = None):
        self.app, self.asr, self.listener, self.tts = app, asr, listener, tts
        self.hooks = hooks or Hooks()
        self.cfg = app.cfg
        self.inbox: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.active_until = 0.0
        self._busy = False
        self.window = float(self.cfg.wake.conversation_window_sec)
        self.sr = int(self.cfg.audio.sample_rate)
        if tts is not None:
            tts.on_play_start = self._on_play_start
            tts.on_play_end = lambda: self.hooks.state(THINKING if self._busy else IDLE)
        app.ctx.confirm = self._confirm

    # ---------- управление ----------
    def submit(self, text: str) -> None:
        """Команда, введённая текстом (имя не нужно)."""
        if text.strip():
            self.inbox.put(text.strip())

    def stop(self) -> None:
        self.stop_event.set()

    def _should_break(self) -> bool:
        return self.stop_event.is_set() or not self.inbox.empty()

    # ---------- речь ----------
    def say(self, text: str) -> None:
        if self.tts:
            self.tts.speak(text)        # текст появится в окне, когда начнёт звучать (_on_play_start)
        else:
            self.hooks.answer(text)

    def _on_play_start(self, text: str) -> None:
        self.hooks.answer(text)
        self.hooks.state(SPEAKING)

    def _wait_speech(self) -> None:
        if self.tts:
            self.tts.wait()

    def prewarm_phrases(self) -> list[str]:
        owner = self.cfg.persona.owner_title
        phrases = [f"Я на месте, {owner}.", "Да, слушаю.", "Делаю громче.", "Делаю тише.", "Переключаю звук.",
                   "Хорошо.", "Следующий трек.", "Предыдущий трек.", "Сворачиваю все окна.", "Открытых задач нет.",
                   "Открываю загрузки.", "Открываю документы.", "Открываю рабочий стол."]
        phrases += [f"Запускаю {a.title}." for a in (getattr(self.cfg, "apps", None) or [])]
        return phrases

    # ---------- подтверждения опасных действий ----------
    def _confirm(self, question: str) -> bool:
        self.say(question)
        self._wait_speech()
        self.hooks.state(LISTENING)
        if self.listener is None:   # режим без микрофона: ждём ответ, набранный текстом
            try:
                answer = self.inbox.get(timeout=20)
            except queue.Empty:
                self.hooks.state(THINKING)
                return False
            self.hooks.heard(answer, True)
            self.hooks.state(THINKING)
            return parse_yes_no(answer) is True
        audio = self.listener.listen(timeout=8, stop=lambda: not self.inbox.empty() or self.stop_event.is_set())
        if not self.inbox.empty():
            answer = self.inbox.get()
        elif audio is None:
            self.hooks.state(THINKING)
            return False
        else:
            answer = self.asr.transcribe(audio)
        self.hooks.heard(answer, True)
        self.hooks.state(THINKING)
        return parse_yes_no(answer) is True

    # ---------- обработка команды ----------
    def _process(self, command: str) -> None:
        self._busy = True
        self.hooks.state(THINKING)
        t0 = time.time()
        try:
            final = self.app.agent.respond(command, on_say=self.say, on_tool=self.hooks.tool)
            t_agent = time.time() - t0
            t1 = time.time()
            if self.tts:
                self._wait_speech()
            else:   # без озвучки показываем «говорящую» позу ровно столько, сколько читается ответ
                self.hooks.state(SPEAKING)
                time.sleep(min(8.0, 0.05 * len(final)))
            self.hooks.status(f"агент {t_agent:.1f} с · дозвучка {time.time() - t1:.1f} с")
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка при выполнении команды")
            self.hooks.answer(f"Что-то пошло не так: {type(e).__name__}: {e}")
        finally:
            self._busy = False
            self.active_until = time.time() + self.window
            self.hooks.state(IDLE)

    # ---------- главный цикл ----------
    def run(self) -> None:
        owner = self.cfg.persona.owner_title
        self.hooks.state(IDLE)
        if self.tts and bool(getattr(self.cfg.tts, "prewarm", True)):
            self.tts.prewarm(self.prewarm_phrases())
        self.say(f"Я на месте, {owner}.")
        self._wait_speech()
        self.hooks.state(IDLE)

        while not self.stop_event.is_set():
            try:
                typed = self.inbox.get_nowait()
            except queue.Empty:
                typed = None
            if typed is not None:
                self.hooks.heard(typed, True)
                self._process(typed)
                continue

            if self.listener is None:   # режим без микрофона: просто ждём текстовые команды
                time.sleep(0.2)
                continue
            audio = self.listener.listen(stop=self._should_break,
                                         on_speech_start=lambda: self.hooks.state(LISTENING))
            if audio is None:
                self.hooks.state(IDLE)
                continue
            if audio.size < MIN_PHRASE_SEC * self.sr:
                self.hooks.state(IDLE)
                continue
            t0 = time.time()
            text = self.asr.transcribe(audio)
            t_asr = time.time() - t0
            log.info("распознано за %.1f с (звук %.1f с): %r", t_asr, audio.size / self.sr, text)
            if not text:
                self.hooks.state(IDLE)
                continue

            woke, cleaned = split_wake(text, list(self.cfg.wake.words), float(self.cfg.wake.fuzzy_ratio))
            addressed = woke or (self.window > 0 and time.time() <= self.active_until)
            self.hooks.heard(text, addressed)
            if not addressed:
                self.hooks.state(IDLE)
                continue
            self.hooks.status(f"распознавание {t_asr:.1f} с")
            command = cleaned if woke else text
            if not command.strip():
                self._busy = True
                self.say("Да, слушаю.")
                self._wait_speech()
                self._busy = False
                self.active_until = time.time() + self.window
                self.hooks.state(IDLE)
                continue
            self._process(command)

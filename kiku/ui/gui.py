"""Окно Кику (Tkinter): аватар по центру, ответ сверху, распознанная речь снизу.

Позы: idle — стоит и улыбается; listening — рука у уха (слышит речь); thinking — задумалась;
speaking — говорит (во время озвучки). Запуск без консоли: pythonw -m kiku gui (или Kiku.exe).
Все тяжёлые модели грузятся в фоне: окно появляется сразу.
"""
import logging
import math
import os
import queue
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..env import ROOT

log = logging.getLogger("kiku.gui")
ASSETS = Path(__file__).parent / "assets"
POSES = ("idle", "listening", "thinking", "speaking")
STATE_LABEL = {"idle": "", "listening": "Слушаю…", "thinking": "Думаю…", "speaking": "Говорю…"}

BG, PANEL, INK, MUTED, ACCENT, ERROR = "#FFF4FA", "#FFFFFF", "#3B2A45", "#9A87A8", "#D96AA7", "#C0392B"
PLACEHOLDER = "Или напишите команду и нажмите Enter…"


# ---------------------------------------------------------------- окно
class KikuWindow:
    def __init__(self, cfg, events: queue.Queue, on_submit, on_close):
        import tkinter as tk
        from tkinter import font as tkfont
        from PIL import Image

        self.tk, self.Image = tk, Image
        self.cfg, self.events, self.on_submit, self.on_close = cfg, events, on_submit, on_close
        self.pose = "idle"
        self.ready = False
        self.sprites = {n: Image.open(ASSETS / f"{n}.png").convert("RGBA") for n in POSES}
        self.photos: dict = {}          # (поза, размер) -> PhotoImage
        self.size = (0, 0)
        self._resize_job = None

        root = self.root = tk.Tk()
        root.title(cfg.persona.name)
        root.configure(bg=BG)
        root.geometry(f"{int(cfg.gui.width)}x{int(cfg.gui.height)}")
        root.minsize(420, 640)
        if getattr(cfg.gui, "always_on_top", False):
            root.attributes("-topmost", True)
        try:
            root.iconbitmap(str(ASSETS / "kiku.ico"))
        except Exception:  # noqa: BLE001
            pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        body = tkfont.Font(family="Segoe UI", size=12)
        small = tkfont.Font(family="Segoe UI", size=11)
        tiny = tkfont.Font(family="Segoe UI", size=9)

        # --- ответ (сверху)
        self.answer = tk.Text(root, height=5, wrap="word", font=body, bg=PANEL, fg=INK, relief="flat", bd=0,
                              padx=14, pady=10, highlightthickness=1, highlightbackground="#EBCFE0",
                              highlightcolor="#EBCFE0", state="disabled", cursor="arrow")
        self.answer.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 6))

        # --- аватар (по центру)
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.img_id = self.canvas.create_image(0, 0, anchor="s")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # --- распознанная речь (снизу)
        self.heard = tk.Label(root, text="", bg=BG, fg=INK, font=small, wraplength=480, justify="center")
        self.heard.grid(row=2, column=0, sticky="ew", padx=18, pady=(6, 0))
        self.status = tk.Label(root, text="Загружаюсь…", bg=BG, fg=MUTED, font=tiny)
        self.status.grid(row=3, column=0, sticky="ew", pady=(2, 4))

        # --- ввод текстом
        row = tk.Frame(root, bg=BG)
        row.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 14))
        row.columnconfigure(0, weight=1)
        self.entry = tk.Entry(row, font=small, relief="flat", bg=PANEL, fg=MUTED, insertbackground=INK,
                              highlightthickness=1, highlightbackground="#EBCFE0", highlightcolor=ACCENT)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=7)
        self.entry.insert(0, PLACEHOLDER)
        self.entry.bind("<FocusIn>", self._entry_focus_in)
        self.entry.bind("<FocusOut>", self._entry_focus_out)
        self.entry.bind("<Return>", self._submit)
        self.entry.config(state="disabled")
        root.bind("<Configure>", self._on_root_resize)

        root.after(33, self._tick)
        root.after(30, self._poll)

    # ---------- ввод
    def _entry_focus_in(self, _e=None):
        if self.entry.get() == PLACEHOLDER:
            self.entry.delete(0, "end")
            self.entry.config(fg=INK)

    def _entry_focus_out(self, _e=None):
        if not self.entry.get():
            self.entry.insert(0, PLACEHOLDER)
            self.entry.config(fg=MUTED)

    def _submit(self, _e=None):
        text = self.entry.get().strip()
        if not self.ready or not text or text == PLACEHOLDER:
            return
        self.entry.delete(0, "end")
        self.on_submit(text)

    # ---------- размеры и картинки
    def _on_root_resize(self, e):
        if e.widget is self.root:
            self.heard.config(wraplength=max(220, e.width - 60))

    def _on_canvas_resize(self, e):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, lambda: self._rebuild(e.width, e.height))

    def _rebuild(self, cw: int, ch: int):
        from PIL import ImageTk
        w0, h0 = self.sprites["idle"].size
        scale = min((cw - 16) / w0, (ch - 12) / h0, 1.0)
        size = (max(int(w0 * scale), 50), max(int(h0 * scale), 80))
        self.size = size
        self.photos = {n: ImageTk.PhotoImage(img.resize(size, self.Image.LANCZOS)) for n, img in self.sprites.items()}
        self.canvas.itemconfig(self.img_id, image=self.photos[self.pose])
        self._base = (cw // 2, ch - 6)
        self.canvas.coords(self.img_id, *self._base)

    # ---------- анимация (лёгкое «дыхание»; во время речи — покачивание)
    def _tick(self):
        try:
            if self.photos and hasattr(self, "_base"):
                t = time.time()
                if self.pose == "speaking":
                    dx, dy = 0.0, -abs(6 * math.sin(2 * math.pi * 2.0 * t))
                elif self.pose == "thinking":
                    dx, dy = 3 * math.sin(2 * math.pi * 0.3 * t), 2 * math.sin(2 * math.pi * 0.5 * t)
                elif self.pose == "listening":
                    dx, dy = 0.0, 2 * math.sin(2 * math.pi * 0.8 * t)
                else:
                    dx, dy = 0.0, 3 * math.sin(2 * math.pi * 0.25 * t)
                self.canvas.coords(self.img_id, self._base[0] + dx, self._base[1] + dy)
        finally:
            self.root.after(33, self._tick)

    # ---------- события из рабочего потока
    def _poll(self):
        try:
            while True:
                self.handle(self.events.get_nowait())
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001
            log.exception("Ошибка обработки события интерфейса")
        finally:
            self.root.after(30, self._poll)

    def set_pose(self, pose: str):
        self.pose = pose
        if pose in POSES and self.photos:
            self.canvas.itemconfig(self.img_id, image=self.photos[pose])

    def set_answer(self, text: str, error: bool = False, append: bool = False):
        a = self.answer
        a.config(state="normal")
        if not append:
            a.delete("1.0", "end")
        elif a.get("1.0", "end-1c"):
            a.insert("end", "\n")
        a.insert("end", text)
        a.config(fg=ERROR if error else INK, state="disabled")
        a.see("end")

    def handle(self, ev: tuple):
        kind = ev[0]
        if kind == "state":
            self.set_pose(ev[1])
            if self.ready:
                self.status.config(text=STATE_LABEL.get(ev[1], ""))
        elif kind == "heard":
            _, text, addressed = ev
            if addressed:
                self.set_answer("")          # новая команда — очищаем прошлый ответ
                self.heard.config(text=f"Вы: {text}", fg=INK)
            else:
                self.heard.config(text=f"(не мне) {text}", fg=MUTED)
        elif kind == "answer":
            self.set_answer(ev[1], append=True)
        elif kind == "status":
            self.status.config(text=ev[1])
        elif kind == "loading":
            self.status.config(text=ev[1])
        elif kind == "ready":
            self.ready = True
            self.entry.config(state="normal")
            self.status.config(text="")
            self.set_pose("idle")
        elif kind == "error":
            self.set_answer(ev[1], error=True)
            self.status.config(text="")

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------- запуск
def _setup_logging() -> Path:
    """Без консоли (pythonw) печатать некуда — всё пишем в data/kiku.log."""
    log_path = ROOT / "data" / "kiku.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logging.getLogger("kiku").setLevel(logging.INFO)
    logging.getLogger("sox").setLevel(logging.ERROR)
    if sys.stdout is None or sys.stderr is None:      # pythonw: у процесса нет консольных потоков
        stream = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    return log_path


def _windows_tweaks() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)                        # чёткая картинка на HiDPI
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("kiku.assistant")   # своя иконка на панели задач
    except Exception:  # noqa: BLE001
        pass


def run_gui(cfg) -> int:
    log_path = _setup_logging()
    _windows_tweaks()
    events: queue.Queue = queue.Queue()
    holder: dict = {}

    def on_submit(text: str):
        s = holder.get("session")
        if s:
            s.submit(text)

    def on_close():
        s = holder.get("session")
        if s:
            s.stop()
        app = holder.get("app")
        try:
            window.root.destroy()
        finally:
            if app:
                try:
                    app.memory.close()
                except Exception:  # noqa: BLE001
                    pass
            os._exit(0)   # не ждём выгрузки CUDA/звуковых потоков

    try:
        window = KikuWindow(cfg, events, on_submit, on_close)
    except Exception as e:  # noqa: BLE001
        log.exception("Не удалось создать окно")
        _fatal(f"Не удалось создать окно: {e}\nПодробности: {log_path}")
        return 1

    def worker():
        from ..app import build_app
        from ..voice.session import Hooks, VoiceSession
        try:
            events.put(("loading", "Загружаю языковую модель…"))
            app = build_app(cfg)
            holder["app"] = app

            asr = listener = tts = None
            try:
                events.put(("loading", "Загружаю распознавание речи…"))
                from ..voice.asr import WhisperASR
                from ..voice.listener import MicListener
                asr = WhisperASR(cfg.asr)
                listener = MicListener(cfg.audio)
            except Exception as e:  # noqa: BLE001
                log.exception("Голосовой ввод недоступен")
                asr = listener = None
                events.put(("answer", f"Микрофон/распознавание недоступны ({type(e).__name__}). Можно писать командами внизу."))
            if cfg.tts.enabled:
                try:
                    events.put(("loading", "Загружаю голос…"))
                    from ..voice.tts import QwenTTS
                    tts = QwenTTS(cfg.tts)
                except Exception as e:  # noqa: BLE001
                    log.exception("Озвучка недоступна")
                    tts = None
                    events.put(("answer", f"Озвучка недоступна ({type(e).__name__}): отвечаю только текстом."))
            events.put(("loading", "Прогреваю языковую модель…"))
            app.agent.warmup()

            hooks = Hooks(
                state=lambda s: events.put(("state", s)),
                heard=lambda t, a: events.put(("heard", t, a)),
                answer=lambda t: events.put(("answer", t)),
                status=lambda s: log.info("тайминг: %s", s),
                tool=lambda tool, args, res: events.put(
                    ("status", f"{'⚙' if res.get('ok') else '✗'} {tool}")),
            )
            session = VoiceSession(app, asr, listener, tts, hooks)
            holder["session"] = session
            events.put(("ready",))
            session.run()
        except Exception as e:  # noqa: BLE001
            log.exception("Критическая ошибка рабочего потока")
            events.put(("error", f"Ошибка: {type(e).__name__}: {e}\nПодробности в {log_path}"))

    threading.Thread(target=worker, daemon=True, name="kiku-worker").start()
    window.run()
    return 0


def _fatal(message: str) -> None:
    """Сообщение об ошибке без консоли."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, message, "Кику", 0x10)
            return
        except Exception:  # noqa: BLE001
            pass
    print(message, file=sys.stderr)

"""Окно Кику (Tkinter): аватар по центру, ответ сверху, распознанная речь снизу, «?» — справка о навыках.

Позы: idle — стоит и улыбается; listening — рука у уха (слышит речь); thinking — задумалась;
speaking — говорит (во время озвучки). Текст ответа появляется, когда Кику начинает его произносить.
Запуск без консоли: pythonw -m kiku gui (или Kiku.exe). Все тяжёлые модели грузятся в фоне: окно появляется сразу.
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
from .skills import help_sections

log = logging.getLogger("kiku.gui")
ASSETS = Path(__file__).parent / "assets"
POSES = ("idle", "listening", "thinking", "speaking")
STATE_LABEL = {"idle": "", "listening": "Слушаю…", "thinking": "Думаю…", "speaking": "Говорю…"}

BG, PANEL, INK, MUTED, ACCENT, ERROR = "#FFF4FA", "#FFFFFF", "#3B2A45", "#9A87A8", "#D96AA7", "#C0392B"
PLACEHOLDER = "Или напишите команду и нажмите Enter…"
BASE_BODY, BASE_SMALL, BASE_TINY = 12, 11, 9      # размеры шрифтов при высоте окна 900 px
MAX_SPRITE_SCALE = 2.5                            # на большом экране Кику увеличивается, но не бесконечно


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
        self.photos: dict = {}          # поза -> PhotoImage
        self.size = (0, 0)
        self._resize_job = None
        self._help = None
        self._fullscreen = False

        root = self.root = tk.Tk()
        root.title(cfg.persona.name)
        root.configure(bg=BG)
        root.geometry(f"{int(cfg.gui.width)}x{int(cfg.gui.height)}")
        root.minsize(420, 640)
        if getattr(cfg.gui, "always_on_top", False):
            root.attributes("-topmost", True)
        if getattr(cfg.gui, "start_maximized", True):
            try:
                root.state("zoomed")            # Windows: на весь экран, но с рамкой и кнопками окна
            except tk.TclError:
                pass
        try:
            root.iconbitmap(str(ASSETS / "kiku.ico"))
        except Exception:  # noqa: BLE001
            pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.bind("<F1>", lambda _e: self.show_help())
        root.bind("<F11>", lambda _e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda _e: self.toggle_fullscreen(False) if self._fullscreen else None)

        self.f_body = tkfont.Font(family="Segoe UI", size=BASE_BODY)
        self.f_small = tkfont.Font(family="Segoe UI", size=BASE_SMALL)
        self.f_tiny = tkfont.Font(family="Segoe UI", size=BASE_TINY)
        self.f_title = tkfont.Font(family="Segoe UI", size=BASE_BODY + 4, weight="bold")

        # --- шапка: имя и кнопка справки
        header = tk.Frame(root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 0))
        header.columnconfigure(0, weight=1)
        tk.Label(header, text=cfg.persona.name, bg=BG, fg=ACCENT, font=self.f_title).grid(row=0, column=0, sticky="w")
        self.help_btn = tk.Button(header, text="?", font=self.f_title, bg=ACCENT, fg="white", activebackground=INK,
                                  activeforeground="white", relief="flat", bd=0, width=2, cursor="hand2",
                                  command=self.show_help)
        self.help_btn.grid(row=0, column=1, sticky="e")

        # --- ответ (сверху)
        self.answer = tk.Text(root, height=5, wrap="word", font=self.f_body, bg=PANEL, fg=INK, relief="flat", bd=0,
                              padx=14, pady=10, highlightthickness=1, highlightbackground="#EBCFE0",
                              highlightcolor="#EBCFE0", state="disabled", cursor="arrow")
        self.answer.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 6))

        # --- аватар (по центру)
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.img_id = self.canvas.create_image(0, 0, anchor="s")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # --- распознанная речь (снизу)
        self.heard = tk.Label(root, text="", bg=BG, fg=INK, font=self.f_small, wraplength=480, justify="center")
        self.heard.grid(row=3, column=0, sticky="ew", padx=18, pady=(6, 0))
        self.status = tk.Label(root, text="Загружаюсь…", bg=BG, fg=MUTED, font=self.f_tiny)
        self.status.grid(row=4, column=0, sticky="ew", pady=(2, 4))

        # --- ввод текстом
        self.row = tk.Frame(root, bg=BG)
        self.row.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 14))
        self.row.columnconfigure(0, weight=1)
        self.entry = tk.Entry(self.row, font=self.f_small, relief="flat", bg=PANEL, fg=MUTED, insertbackground=INK,
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

    # ---------- полноэкранный режим и справка
    def toggle_fullscreen(self, on: bool | None = None):
        self._fullscreen = (not self._fullscreen) if on is None else on
        self.root.attributes("-fullscreen", self._fullscreen)

    def show_help(self):
        """Окно «Что умеет Кику» (текст — kiku/ui/skills.py)."""
        tk = self.tk
        if self._help is not None and self._help.winfo_exists():
            self._help.lift()
            self._help.focus_force()
            return
        top = self._help = tk.Toplevel(self.root)
        top.title(f"Что умеет {self.cfg.persona.name}")
        top.configure(bg=BG)
        top.geometry("680x760")
        top.transient(self.root)
        top.bind("<Escape>", lambda _e: top.destroy())
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        box = tk.Text(top, wrap="word", font=self.f_small, bg=PANEL, fg=INK, relief="flat", bd=0, padx=22, pady=16,
                      highlightthickness=0, cursor="arrow")
        bar = tk.Scrollbar(top, command=box.yview)
        box.configure(yscrollcommand=bar.set)
        box.grid(row=0, column=0, sticky="nsew", padx=(14, 0), pady=14)
        bar.grid(row=0, column=1, sticky="ns", pady=14, padx=(0, 6))
        box.tag_configure("h", font=self.f_title, foreground=ACCENT, spacing1=14, spacing3=4)
        box.tag_configure("ex", foreground=INK, lmargin1=14, lmargin2=14, spacing3=2)
        box.tag_configure("note", foreground=MUTED, lmargin1=14, lmargin2=14, spacing3=2)
        for title, lines in help_sections(self.cfg):
            box.insert("end", title + "\n", "h")
            for line in lines:
                box.insert("end", line + "\n", "ex" if line.startswith("«") else "note")
        box.config(state="disabled")
        tk.Button(top, text="Закрыть", command=top.destroy, bg=ACCENT, fg="white", activebackground=INK,
                  activeforeground="white", relief="flat", bd=0, padx=18, pady=6, cursor="hand2",
                  font=self.f_small).grid(row=1, column=0, columnspan=2, pady=(0, 14))

    # ---------- размеры и картинки
    def _on_root_resize(self, e):
        if e.widget is not self.root:
            return
        k = min(max(e.height / 900, 0.9), 1.9)     # шрифты растут вместе с окном
        for font, base in ((self.f_body, BASE_BODY), (self.f_small, BASE_SMALL), (self.f_tiny, BASE_TINY),
                           (self.f_title, BASE_BODY + 4)):
            size = round(base * k)
            if font.cget("size") != size:
                font.configure(size=size)
        pad = max(18, int(e.width * 0.12))         # на широком экране текст не растягивается на всю ширину
        self.answer.grid_configure(padx=pad)
        self.row.grid_configure(padx=pad)
        self.heard.grid_configure(padx=pad)
        self.heard.config(wraplength=max(220, e.width - 2 * pad - 20))

    def _on_canvas_resize(self, e):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, lambda: self._rebuild(e.width, e.height))

    def _rebuild(self, cw: int, ch: int):
        from PIL import ImageTk
        w0, h0 = self.sprites["idle"].size
        scale = min((cw - 16) / w0, (ch - 12) / h0, MAX_SPRITE_SCALE)
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
            if self.ready and ev[1] != "idle":
                self.status.config(text=STATE_LABEL.get(ev[1], ""))
        elif kind == "heard":
            _, text, addressed = ev
            if addressed:
                self.set_answer("")          # новая команда — очищаем прошлый ответ
                self.heard.config(text=f"Вы: {text}", fg=INK)
            else:
                self.heard.config(text=f"(не мне) {text}", fg=MUTED)
        elif kind == "notice":                   # Кику заговорила сама: напоминание, сводка, забота о перерыве
            self.set_answer("")
            self.heard.config(text=ev[1], fg=ACCENT)
        elif kind == "answer":
            self.set_answer(ev[1], append=True)
        elif kind in ("status", "loading"):
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
        from .. import devices
        from ..app import build_app
        from ..cuda_init import init_cuda_first
        from ..voice.session import Hooks, VoiceSession
        try:
            # ПОРЯДОК ВАЖЕН: torch и его cuDNN должны загрузиться раньше, чем faster-whisper (иначе на Windows
            # падает «Could not load symbol cudnnGetLibConfig»). Подробности — kiku/cuda_init.py.
            events.put(("loading", "Инициализирую видеокарту…"))
            log.info("CUDA: %s", init_cuda_first())
            events.put(("loading", "Загружаю языковую модель…"))
            app = build_app(cfg)
            holder["app"] = app

            asr = listener = tts = None
            if cfg.tts.enabled:
                try:
                    events.put(("loading", "Загружаю голос…"))
                    from ..voice.tts import build_tts
                    tts = build_tts(cfg.tts)
                except Exception as e:  # noqa: BLE001
                    log.exception("Озвучка недоступна")
                    tts = None
                    events.put(("answer", f"Озвучка недоступна ({type(e).__name__}): отвечаю только текстом."))
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
            events.put(("loading", "Прогреваю языковую модель…"))
            app.agent.warmup()

            for line in devices.report(asr, tts, app.agent.llm):
                log.info("устройства — %s", line)

            hooks = Hooks(
                state=lambda s: events.put(("state", s)),
                heard=lambda t, a: events.put(("heard", t, a)),
                notice=lambda title: events.put(("notice", title)),
                answer=lambda t: events.put(("answer", t)),
                status=lambda s: log.info("тайминг: %s", s),
                tool=lambda tool, args, res: events.put(
                    ("status", f"{'⚙' if res.get('ok') else '✗'} {tool}")),
            )
            session = VoiceSession(app, asr, listener, tts, hooks)
            holder["session"] = session
            events.put(("ready",))
            events.put(("status", devices.summary(asr, tts, app.agent.llm)))
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

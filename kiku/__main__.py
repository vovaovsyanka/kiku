"""Точка входа: python -m kiku [chat|voice|gui|web] [--dry-run] [--config путь]"""
import argparse
import logging
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="kiku", description="Кику — локальная голосовая помощница")
    ap.add_argument("mode", nargs="?", default="chat", choices=["chat", "voice", "web", "gui"],
                    help="chat — текст (по умолчанию), voice — голос в консоли, gui — окно с Кику, web — браузер (экспериментально)")
    ap.add_argument("--config", help="путь к config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="инструменты ничего не делают, только пишут в лог")
    ap.add_argument("--tts", action="store_true", help="озвучивать ответы в режиме chat/web")
    ap.add_argument("--no-tts", action="store_true", help="не озвучивать в режиме voice")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from .config import load_config
    from .env import setup_environment

    cfg = load_config(args.config)
    setup_environment(offline=bool(cfg.security.offline))   # до импорта любых HF-библиотек
    if args.mode != "gui":   # у окна своё логирование в файл (без консоли печатать некуда)
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("kiku").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    logging.getLogger("sox").setLevel(logging.ERROR)   # пакет sox ругается на отсутствие SoX; нам он не нужен

    if args.mode == "gui":
        from .ui.gui import run_gui
        return run_gui(cfg)

    from .app import build_app

    print("Загружаю языковую модель (первый запуск может занять минуту)…")
    app = build_app(cfg, dry_run=True if args.dry_run else None)

    tts = None
    want_tts = cfg.tts.enabled and ((args.mode == "voice" and not args.no_tts) or args.tts)
    if want_tts:
        from .voice.tts import QwenTTS
        tts = QwenTTS(cfg.tts)

    try:
        if args.mode == "chat":
            from .ui.cli import run_chat
            run_chat(app, tts)
        elif args.mode == "voice":
            from .voice.asr import WhisperASR
            from .voice.listener import MicListener
            from .voice.loop import run_voice
            asr = WhisperASR(cfg.asr)
            listener = MicListener(cfg.audio)
            run_voice(app, asr, listener, tts)
        else:
            from .ui.web import run_web
            run_web(app, tts)
    finally:
        app.memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

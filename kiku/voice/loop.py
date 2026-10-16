"""Консольный голосовой режим (python -m kiku voice). Окно с Кику — python -m kiku gui."""
from .session import Hooks, VoiceSession


def run_voice(app, asr, listener, tts=None) -> None:
    cfg = app.cfg
    name = cfg.persona.name
    window = float(cfg.wake.conversation_window_sec)

    def heard(text: str, addressed: bool) -> None:
        if addressed:
            print(f"Вы: {text}")

    hooks = Hooks(
        heard=heard,
        answer=lambda t: print(f"{name}: {t}"),
        status=lambda s: print(f"   [время] {s}"),
        tool=lambda tool, args, res: print(f"   [инструмент] {tool}({args}) -> {'ok' if res.get('ok') else 'ОШИБКА'}"),
    )
    session = VoiceSession(app, asr, listener, tts, hooks)
    if window > 0:
        print(f"\n{name} слушает. Имя нужно в первой реплике, затем {window:.0f} с можно без него. Ctrl+C — выход.\n")
    else:
        print(f"\n{name} слушает. Каждая команда начинается с имени: «{name}, открой загрузки». Ctrl+C — выход.\n")
    print("Прогреваю языковую модель…")
    app.agent.warmup()
    try:
        session.run()
    except KeyboardInterrupt:
        print("\nДо встречи!")

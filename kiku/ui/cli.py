"""Текстовый режим: удобно для отладки агента без микрофона и TTS."""
import logging

log = logging.getLogger("kiku.cli")


def run_chat(app, tts=None) -> None:
    def confirm(question: str) -> bool:
        ans = input(f"\n  ? {question} [y/N]: ").strip().lower()
        return ans in ("y", "yes", "д", "да")

    app.ctx.confirm = confirm
    name = app.cfg.persona.name
    print(f"\n{name} готова. Команды: /tools — список инструментов, /reset — забыть диалог, /quit — выход.")
    if app.ctx.dry_run:
        print("(режим dry-run: инструменты ничего реально не делают)")
    print()

    def on_say(text: str) -> None:
        print(f"{name}: {text}")
        if tts:
            tts.speak(text)

    def on_tool(tool, args, result) -> None:
        mark = "ok" if result.get("ok") else "ОШИБКА"
        print(f"   [инструмент] {tool}({args}) -> {mark}")

    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/tools":
            print(app.registry.catalog())
            continue
        if text == "/reset":
            app.agent.reset()
            print("(диалог сброшен)")
            continue
        app.agent.respond(text, on_say=on_say, on_tool=on_tool)

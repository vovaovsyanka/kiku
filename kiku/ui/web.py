"""Веб-интерфейс (ЭКСПЕРИМЕНТАЛЬНО): чат + индикатор состояния. Задел под аватар.

Слушает только 127.0.0.1. События, которые получает страница по WebSocket:
  {"type": "state", "state": "idle" | "thinking" | "speaking"}
  {"type": "say",   "text": "..."}
  {"type": "tool",  "name": "...", "ok": true}
Именно сюда позже подключается Live2D / VRM-аватар: он реагирует на state и say.
Подтверждения опасных действий в вебе пока не поддерживаются (инструменты их отклоняют).
"""
import asyncio
import re
from pathlib import Path

STATIC = Path(__file__).parent / "static"
_LOCAL_ORIGIN = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$")


def create_web_app(app, tts=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse

    web = FastAPI(title="Kiku")
    lock = asyncio.Lock()
    app.ctx.confirm = lambda question: False  # веб-подтверждения — в планах

    @web.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @web.websocket("/ws")
    async def ws(sock: WebSocket):
        origin = sock.headers.get("origin", "")
        if origin and not _LOCAL_ORIGIN.match(origin):  # защита от чужих сайтов в браузере
            await sock.close(code=1008)
            return
        await sock.accept()
        loop = asyncio.get_running_loop()

        def push(obj):
            asyncio.run_coroutine_threadsafe(sock.send_json(obj), loop)

        try:
            while True:
                msg = await sock.receive_json()
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                async with lock:
                    await sock.send_json({"type": "state", "state": "thinking"})

                    def on_say(t):
                        push({"type": "say", "text": t})
                        if tts:
                            push({"type": "state", "state": "speaking"})
                            tts.speak(t)

                    def on_tool(name, args, result):
                        push({"type": "tool", "name": name, "ok": bool(result.get("ok"))})

                    await asyncio.to_thread(app.agent.respond, text, on_say, on_tool)
                    if tts:
                        await asyncio.to_thread(tts.wait)
                    await sock.send_json({"type": "state", "state": "idle"})
        except WebSocketDisconnect:
            pass

    return web


def run_web(app, tts=None, port: int = 8765) -> None:
    import uvicorn
    print(f"Откройте в браузере: http://127.0.0.1:{port}")
    uvicorn.run(create_web_app(app, tts), host="127.0.0.1", port=port, log_level="warning")

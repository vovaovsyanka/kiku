"""Озвучка: движок синтеза (engines.py) + конвейер (speech.py) + кэш готовых фраз на диске.

Озвучивается ВЕСЬ текст ответа: он режется по предложениям, куски синтезируются по очереди, а воспроизведение
стартует, как только оставшаяся часть успеет досинтезироваться (см. speech.py). Движок выбирается в config.yaml → tts.engine.
"""
import hashlib
import logging
import re
import threading
import time

from ..config import resolve_path
from .speech import SpeechPipeline

log = logging.getLogger("kiku.tts")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Куски по границам предложений, каждый не длиннее max_chars (если предложение само не длиннее)."""
    chunks: list[str] = []
    cur = ""
    for s in split_sentences(text) or [text.strip()]:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def shorten_for_speech(text: str, max_chars: int) -> str:
    """Необязательное ограничение длины озвучки (tts.max_spoken_chars > 0). По умолчанию 0 — озвучивается всё."""
    text = " ".join((text or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    out = ""
    for sentence in split_sentences(text):
        if out and len(out) + 1 + len(sentence) > max_chars:
            break
        out = f"{out} {sentence}".strip()
        if len(out) >= max_chars:
            break
    if len(out) > max_chars:
        cut = out[:max_chars]
        i = max(cut.rfind(","), cut.rfind(" "))
        out = (cut[:i] if i > max_chars * 0.5 else cut).rstrip(" ,;:-") + "."
    return out


class _DiskCache:
    def __init__(self, directory, voice_id):
        self.dir, self.voice_id = directory, voice_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, text: str):
        h = hashlib.sha1(f"{self.voice_id}|{text}".encode("utf-8")).hexdigest()[:24]
        return self.dir / f"{h}.wav"

    def get(self, text: str):
        import soundfile as sf
        p = self.path(text)
        if not p.exists():
            return None
        wav, sr = sf.read(str(p), dtype="float32")
        return wav, int(sr)

    def put(self, text: str, wav, sr: int) -> None:
        import soundfile as sf
        sf.write(str(self.path(text)), wav, sr)


class Speaker:
    """Голос Кику. on_play_start(text) / on_play_end() — колбэки для интерфейса."""
    on_play_start = None
    on_play_end = None

    def __init__(self, cfg, engine):
        self.cfg, self.engine = cfg, engine
        self.max_spoken_chars = int(cfg.max_spoken_chars)
        self.cache = _DiskCache(resolve_path(cfg.cache_dir), engine.voice_id) if cfg.cache else None
        self.pipe = SpeechPipeline(
            synth_fn=engine.synth, play_fn=self._play, split_fn=split_chunks,
            chunk_chars=int(cfg.chunk_chars), cache=self.cache, initial_rtf=engine.initial_rtf,
            on_start=lambda text: self.on_play_start(text) if self.on_play_start else None,
            on_end=lambda: self.on_play_end() if self.on_play_end else None,
            stop_fn=self._stop_audio)

    # ---- диагностика ----
    def device_info(self) -> str:
        return self.engine.device_info()

    def device_label(self) -> str:
        return self.engine.device_label()

    def synth(self, text: str):
        return self.engine.synth(text)

    # ---- звук ----
    @staticmethod
    def _play(wav, sr) -> None:
        import sounddevice as sd
        sd.play(wav, sr)
        sd.wait()

    @staticmethod
    def _stop_audio() -> None:
        import sounddevice as sd
        sd.stop()

    # ---- публичный интерфейс ----
    def speak(self, text: str) -> None:
        self.pipe.submit(shorten_for_speech(text, self.max_spoken_chars))

    def wait(self) -> None:
        """Дождаться, пока всё поставленное в очередь будет произнесено."""
        self.pipe.wait()

    def stop(self) -> None:
        self.pipe.stop()

    def prewarm(self, phrases) -> None:
        """В простое заранее синтезирует типовые фразы в кэш, чтобы потом они звучали мгновенно."""
        if self.cache is None:
            return

        def work():
            for phrase in phrases:
                try:
                    if self.cache.get(phrase) is not None:
                        continue
                    while not self.pipe.idle():      # не мешаем живым ответам
                        time.sleep(0.5)
                    self._synth_to_cache(phrase)
                except Exception:  # noqa: BLE001
                    log.exception("prewarm: не удалось озвучить %r", phrase)
                    return
        threading.Thread(target=work, daemon=True, name="tts-prewarm").start()

    def _synth_to_cache(self, text: str) -> None:
        from .speech import join_audio
        batch = [self.engine.synth(chunk) for chunk in split_chunks(text, int(self.cfg.chunk_chars))]
        if batch:
            self.cache.put(text, *join_audio(batch))


def build_tts(cfg) -> Speaker:
    """Создаёт голос по config.yaml → tts. Модели грузятся здесь (долго)."""
    from .engines import QwenEngine, SileroEngine
    engine_name = str(cfg.engine).lower()
    if engine_name == "silero":
        engine = SileroEngine(cfg.silero, pronounce=vars(cfg.pronounce) if hasattr(cfg, "pronounce") else {})
    elif engine_name == "qwen":
        engine = QwenEngine(cfg.qwen, language=cfg.language)
    else:
        raise ValueError(f"tts.engine: «{cfg.engine}» не поддерживается (silero | qwen)")
    return Speaker(cfg, engine)

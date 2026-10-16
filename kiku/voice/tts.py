"""Озвучка: Qwen3-TTS (модели с Hugging Face, локально).

Режимы:
  clone  — Qwen3-TTS-12Hz-0.6B-Base, голос берётся из voices/kiku_ref.wav (+ .txt с текстом записи)
  custom — Qwen3-TTS-12Hz-0.6B-CustomVoice, готовый тембр (speaker)

Как избегается «рваная» речь:
  * реплика синтезируется ЦЕЛИКОМ и проигрывается одним куском (никаких пауз между предложениями);
    длинные реплики режутся на крупные части, но играют только после того, как готовы все части;
  * готовые фразы кэшируются на диск (data/tts_cache): повторяющиеся реплики звучат мгновенно.
"""
import hashlib
import logging
import queue
import re
import threading
import time

import numpy as np

from ..config import resolve_path

log = logging.getLogger("kiku.tts")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Крупные части по границам предложений, каждая не длиннее max_chars (если возможно)."""
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
    """Озвучиваем только начало ответа (целые предложения): синтез на этой машине медленнее реального времени,
    поэтому каждая лишняя секунда звука стоит нескольких секунд ожидания. На экране показывается весь текст."""
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
    if len(out) > max_chars:   # одно очень длинное предложение — режем по запятой или пробелу
        cut = out[:max_chars]
        i = max(cut.rfind(","), cut.rfind(" "))
        out = (cut[:i] if i > max_chars * 0.5 else cut).rstrip(" ,;:-") + "."
    return out


class QwenTTS:
    on_play_start = None   # колбэки для интерфейса: звук начал / закончил играть
    on_play_end = None

    def __init__(self, cfg):
        import torch
        import transformers
        from qwen_tts import Qwen3TTSModel

        transformers.logging.set_verbosity_error()   # убирает шум "Setting pad_token_id..."
        self.cfg = cfg
        self._torch = torch
        device, dtype_name = cfg.device, cfg.dtype
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA недоступна (в torch нет CUDA-сборки или нет драйвера) — TTS работает на CPU и будет ОЧЕНЬ медленным. "
                        "Проверьте: python scripts/check_env.py")
            device, dtype_name = "cpu", "float32"
        elif str(device).startswith("cuda") and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            log.warning("Эта видеокарта не поддерживает bfloat16 аппаратно — переключаюсь на float16")
            dtype_name = "float16"
        dtype = getattr(torch, dtype_name)
        repo = cfg.base_repo if cfg.mode == "clone" else cfg.custom_repo

        # Передаём ЛОКАЛЬНУЮ папку, а не имя репозитория: transformers при загрузке токенизатора
        # делает сетевой запрос model_info(), который в офлайн-режиме падает.
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(repo_id=repo, local_files_only=True)
        except Exception as e:  # noqa: BLE001
            raise FileNotFoundError(
                f"Модель {repo} не найдена локально. Скачайте: python scripts/download_models.py --only tts") from e

        kwargs = dict(device_map=device, dtype=dtype)
        log.info("Загружаю TTS %s (device=%s, dtype=%s)", repo, device, dtype_name)
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_path, attn_implementation=cfg.attn_implementation, **kwargs)
        except (ValueError, ImportError, TypeError) as e:
            log.warning("attn_implementation=%s не подошёл (%s), загружаю без него", cfg.attn_implementation, e)
            self.model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
        print(f"[TTS] модель на устройстве {device} ({dtype_name})")

        self.clone_prompt = None
        voice_key = str(cfg.speaker)
        if cfg.mode == "clone":
            ref_audio = resolve_path(cfg.ref_audio)
            ref_text_file = resolve_path(cfg.ref_text_file)
            if not ref_audio.exists() or not ref_text_file.exists():
                raise FileNotFoundError(
                    "Нет референсного голоса. Создайте его: python scripts/design_voice.py "
                    f"(ожидаются {ref_audio} и {ref_text_file}) или переключите tts.mode на 'custom'.")
            self.clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio), ref_text=ref_text_file.read_text(encoding="utf-8").strip())
            st = ref_audio.stat()
            voice_key = f"{st.st_mtime_ns}_{st.st_size}"

        self.chunk_chars = int(getattr(cfg, "chunk_chars", 400))
        self.max_spoken_chars = int(getattr(cfg, "max_spoken_chars", 160))
        self.cache_dir = None
        if getattr(cfg, "cache", True):
            self.cache_dir = resolve_path(getattr(cfg, "cache_dir", "data/tts_cache"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_id = f"{cfg.mode}|{repo}|{voice_key}|{cfg.language}"

        self._tq: queue.Queue = queue.Queue()
        self._aq: queue.Queue = queue.Queue()
        threading.Thread(target=self._synth_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    # ---- синтез ----
    def synth(self, text: str):
        t0 = time.time()
        with self._torch.inference_mode():
            if self.clone_prompt is not None:
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=self.cfg.language, voice_clone_prompt=self.clone_prompt)
            else:
                wavs, sr = self.model.generate_custom_voice(
                    text=text, language=self.cfg.language, speaker=self.cfg.speaker)
        wav = wavs[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        took, dur = time.time() - t0, wav.size / float(sr)
        log.info("TTS: %.1f с звука за %.1f с (RTF %.2f; меньше 1.0 — быстрее реального времени)",
                 dur, took, took / max(dur, 1e-6))
        return wav, int(sr)

    def _cache_path(self, text: str):
        h = hashlib.sha1(f"{self.voice_id}|{text}".encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.wav"

    def synth_text(self, text: str):
        """Вся реплика одним куском; с дисковым кэшем."""
        import soundfile as sf
        path = self._cache_path(text) if self.cache_dir else None
        if path is not None and path.exists():
            wav, sr = sf.read(str(path), dtype="float32")
            log.info("TTS: из кэша")
            return wav, int(sr)
        parts, sr = [], 0
        for chunk in split_chunks(text, self.chunk_chars):
            wav, sr = self.synth(chunk)
            parts += [wav, np.zeros(int(0.12 * sr), dtype=np.float32)]
        wav = np.concatenate(parts[:-1]) if len(parts) > 1 else parts[0]
        if path is not None:
            sf.write(str(path), wav, sr)
        return wav, sr

    def _synth_worker(self):
        while True:
            text = self._tq.get()
            try:
                self._aq.put(self.synth_text(text))
            except Exception:  # noqa: BLE001
                log.exception("Ошибка синтеза речи")
            finally:
                self._tq.task_done()

    def _play_worker(self):
        import sounddevice as sd
        while True:
            wav, sr = self._aq.get()
            try:
                if self.on_play_start:
                    self.on_play_start()
                sd.play(wav, sr)
                sd.wait()
            except Exception:  # noqa: BLE001
                log.exception("Ошибка воспроизведения")
            finally:
                try:
                    if self.on_play_end:
                        self.on_play_end()
                finally:
                    self._aq.task_done()

    # ---- публичный интерфейс ----
    def speak(self, text: str) -> None:
        text = shorten_for_speech(text, self.max_spoken_chars)
        if text:
            self._tq.put(text)

    def prewarm(self, phrases) -> None:
        """В простое заранее синтезирует типовые фразы в кэш (на диск), чтобы потом они звучали мгновенно."""
        if self.cache_dir is None:
            return

        def work():
            for phrase in phrases:
                try:
                    if self._cache_path(phrase).exists():
                        continue
                    while self._tq.unfinished_tasks or self._aq.unfinished_tasks:   # не мешаем живым ответам
                        time.sleep(0.5)
                    self.synth_text(phrase)
                except Exception:  # noqa: BLE001
                    log.exception("prewarm: не удалось озвучить %r", phrase)
                    return
        threading.Thread(target=work, daemon=True).start()

    def wait(self) -> None:
        """Дождаться, пока всё поставленное в очередь будет произнесено."""
        self._tq.join()
        self._aq.join()

    def stop(self) -> None:
        import sounddevice as sd
        for q in (self._tq, self._aq):
            try:
                while True:
                    q.get_nowait()
                    q.task_done()
            except queue.Empty:
                pass
        sd.stop()

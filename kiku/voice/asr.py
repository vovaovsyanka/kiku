"""Распознавание речи: faster-whisper (модель с Hugging Face, работает локально)."""
import logging
import os
import re

import numpy as np

log = logging.getLogger("kiku.asr")

_JUNK = ("субтитры", "dimatorzok", "продолжение следует", "спасибо за просмотр",
         "редактор субтитров", "корректор", "подписывайтесь")


class WhisperASR:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        device, ctype = cfg.device, cfg.compute_type
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        if ctype == "auto":
            ctype = "int8_float16" if device == "cuda" else "int8"
        self.device, self.compute_type = device, ctype
        log.info("Загружаю ASR %s (%s, %s)", cfg.model, device, ctype)
        self.model = WhisperModel(
            cfg.model, device=device, compute_type=ctype,
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        )

    def device_info(self) -> str:
        real = getattr(getattr(self.model, "model", None), "device", None)   # ctranslate2: реальное устройство
        return f"{real or self.device} ({self.compute_type})"

    def device_label(self) -> str:
        return str(getattr(getattr(self.model, "model", None), "device", None) or self.device)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.cfg.language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.cfg.initial_prompt or None,
            vad_filter=False,
            no_speech_threshold=float(self.cfg.no_speech_threshold),
        )
        parts = []
        for s in segments:
            if s.no_speech_prob > float(self.cfg.no_speech_threshold) and s.avg_logprob < -1.0:
                continue
            parts.append(s.text.strip())
        text = " ".join(parts).strip()
        if not text or any(j in text.lower() for j in _JUNK):
            return ""
        if echoes_prompt(text, self.cfg.initial_prompt or ""):
            return ""
        return text


def _squash(s: str) -> str:
    return re.sub(r"\W+", "", s.lower().replace("ё", "е"))


def echoes_prompt(text: str, prompt: str) -> bool:
    """Whisper на шуме иногда «пересказывает» подсказку. Отбрасываем только длинный кусок подсказки:
    короткое «Кику» или «Какие у меня задачи?» — это настоящие реплики, они тоже есть в подсказке."""
    t, p = _squash(text), _squash(prompt)
    return bool(p) and len(t) >= 0.5 * len(p) and t in p

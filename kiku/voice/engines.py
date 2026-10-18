"""Движки синтеза речи. Общий интерфейс:

    synth(text) -> (wav: float32, sample_rate)     один кусок текста
    voice_id: str                                    для ключа кэша (меняется голос — кэш не используется)
    initial_rtf: float                               ожидаемая медлительность (секунд работы на секунду звука)
    device_info() / device_label()                   где лежит модель

Silero v5 — быстрый (RTF ~0.05–0.2 на CPU), родной русский, ударения расставляет сам. Голоса фиксированные.
Qwen3-TTS — клонирует любой голос из voices/kiku_ref.wav, но на слабой видеокарте медленный (RTF > 2).
Тяжёлые библиотеки импортируются внутри конструкторов, поэтому модуль безопасно импортировать без torch.
"""
import logging
import re
import time
from xml.sax.saxutils import escape

import numpy as np

from ..config import resolve_path
from .ru_text import speakable

log = logging.getLogger("kiku.tts")

_ALLOWED = re.compile(r"[^а-яё0-9\s.,!?:;—–\-…«»\"']", re.I)
SILERO_SPEAKERS = ("xenia", "baya", "kseniya", "aidar", "eugene")


def silero_model_path(cfg):
    return resolve_path(cfg.model_dir) / f"{cfg.model}.pt"


def silero_model_url(cfg) -> str:
    return f"https://models.silero.ai/models/tts/ru/{cfg.model}.pt"


class SileroEngine:
    """Silero TTS v5 (ru). Модель — один файл models/silero/<model>.pt, скачивается scripts/download_models.py."""
    initial_rtf = 0.3

    def __init__(self, cfg, pronounce: dict | None = None):
        import torch
        import torch.package

        self.cfg, self._torch = cfg, torch
        path = silero_model_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"Нет модели Silero: {path}. Скачайте: python scripts/download_models.py --only tts")
        self.device = str(cfg.device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA недоступна — Silero работает на CPU")
            self.device = "cpu"
        torch.set_num_threads(max(1, int(getattr(cfg, "threads", 4))))
        log.info("Загружаю Silero %s (%s)", cfg.model, self.device)
        self.model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        self.model.to(self.device)
        self.speaker = str(cfg.speaker)
        if self.speaker not in getattr(self.model, "speakers", [self.speaker]):
            raise ValueError(f"У модели {cfg.model} нет голоса «{self.speaker}». Доступны: {', '.join(self.model.speakers)}")
        self.sample_rate = int(cfg.sample_rate)
        self.pronounce = {k.lower(): v for k, v in (pronounce or {}).items()}
        self.voice_id = f"silero|{cfg.model}|{self.speaker}|{cfg.pitch}|{cfg.rate}|{self.sample_rate}"
        self.synth("Привет.")    # первый вызов заметно медленнее остальных — прогреваем заранее

    def prepare(self, text: str) -> str:
        """Всё, что Silero не умеет читать (латиница, цифры, смайлы), приводим к русским словам или убираем."""
        for src, dst in self.pronounce.items():
            text = re.sub(re.escape(src), dst, text, flags=re.I)
        text = speakable(text)
        text = _ALLOWED.sub(" ", text.replace("\n", ". "))
        return re.sub(r"\s+", " ", text).strip()

    def _ssml(self, text: str) -> str | None:
        pitch, rate = str(self.cfg.pitch), str(self.cfg.rate)
        if pitch == rate == "medium":
            return None
        return f'<speak><prosody pitch="{pitch}" rate="{rate}">{escape(text)}</prosody></speak>'

    def synth(self, text: str):
        text = self.prepare(text)
        if not re.search(r"\w", text):
            return np.zeros(int(0.2 * self.sample_rate), dtype=np.float32), self.sample_rate
        ssml = self._ssml(text)
        t0 = time.time()
        with self._torch.inference_mode():
            if ssml:
                wav = self.model.apply_tts(ssml_text=ssml, speaker=self.speaker, sample_rate=self.sample_rate)
            else:
                wav = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
        wav = wav.detach().cpu().numpy().astype(np.float32).squeeze()
        took, dur = time.time() - t0, wav.size / float(self.sample_rate)
        log.info("TTS: %.1f с звука за %.2f с (RTF %.2f)", dur, took, took / max(dur, 1e-6))
        return wav, self.sample_rate

    def device_info(self) -> str:
        return f"Silero {self.cfg.model}, голос {self.speaker}, {self.device}"

    def device_label(self) -> str:
        return self.device


class QwenEngine:
    """Qwen3-TTS: clone — голос из voices/kiku_ref.wav (+ .txt с текстом записи); custom — готовый тембр."""
    initial_rtf = 2.5

    def __init__(self, cfg, language: str = "Russian"):
        import torch
        import transformers
        from qwen_tts import Qwen3TTSModel

        transformers.logging.set_verbosity_error()   # убирает шум "Setting pad_token_id..."
        self.cfg, self.language, self._torch = cfg, language, torch
        device, dtype_name = cfg.device, cfg.dtype
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA недоступна (в torch нет CUDA-сборки или нет драйвера) — Qwen3-TTS работает на CPU и будет ОЧЕНЬ медленным. "
                        "Проверьте: python scripts/check_env.py")
            device, dtype_name = "cpu", "float32"
        elif str(device).startswith("cuda") and dtype_name != "bfloat16":
            log.warning("Qwen3-TTS в %s ломается (NaN в сэмплировании, CUDA assert на этой видеокарте) — использую bfloat16", dtype_name)
            dtype_name = "bfloat16"
        self.device_name, self.dtype_name = device, dtype_name
        repo = cfg.base_repo if cfg.mode == "clone" else cfg.custom_repo

        # Передаём ЛОКАЛЬНУЮ папку, а не имя репозитория: transformers при загрузке токенизатора
        # делает сетевой запрос model_info(), который в офлайн-режиме падает.
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(repo_id=repo, local_files_only=True)
        except Exception as e:  # noqa: BLE001
            raise FileNotFoundError(
                f"Модель {repo} не найдена локально. Скачайте: python scripts/download_models.py --only tts") from e

        kwargs = dict(device_map=device, dtype=getattr(torch, dtype_name))
        log.info("Загружаю Qwen3-TTS %s (device=%s, dtype=%s)", repo, device, dtype_name)
        try:
            self.model = Qwen3TTSModel.from_pretrained(model_path, attn_implementation=cfg.attn_implementation, **kwargs)
        except (ValueError, ImportError, TypeError) as e:
            log.warning("attn_implementation=%s не подошёл (%s), загружаю без него", cfg.attn_implementation, e)
            self.model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)

        self.clone_prompt = None
        voice_key = str(cfg.speaker)
        if cfg.mode == "clone":
            ref_audio, ref_text_file = resolve_path(cfg.ref_audio), resolve_path(cfg.ref_text_file)
            if not ref_audio.exists() or not ref_text_file.exists():
                raise FileNotFoundError(
                    "Нет референсного голоса. Создайте его: python scripts/design_voice.py "
                    f"(ожидаются {ref_audio} и {ref_text_file}) или переключите tts.qwen.mode на 'custom'.")
            self.clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio), ref_text=ref_text_file.read_text(encoding="utf-8").strip())
            st = ref_audio.stat()
            voice_key = f"{st.st_mtime_ns}_{st.st_size}"
        self.voice_id = f"qwen|{cfg.mode}|{repo}|{voice_key}|{language}"

    def synth(self, text: str):
        t0 = time.time()
        with self._torch.inference_mode():
            if self.clone_prompt is not None:
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=self.language, voice_clone_prompt=self.clone_prompt)
            else:
                wavs, sr = self.model.generate_custom_voice(text=text, language=self.language, speaker=self.cfg.speaker)
        wav = wavs[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        took, dur = time.time() - t0, wav.size / float(sr)
        log.info("TTS: %.1f с звука за %.1f с (RTF %.2f)", dur, took, took / max(dur, 1e-6))
        return wav, int(sr)

    def device_info(self) -> str:
        try:
            p = next(self.model.model.parameters())
            where = f"параметры модели на {p.device}, {str(p.dtype).replace('torch.', '')}"
        except Exception:  # noqa: BLE001
            where = f"запрошено {self.device_name}, {self.dtype_name}"
        mem = ""
        if self._torch.cuda.is_available():
            mem = f", занято видеопамяти процессом {self._torch.cuda.memory_allocated() / 2**30:.1f} ГБ"
        return "Qwen3-TTS: " + where + mem

    def device_label(self) -> str:
        try:
            return str(next(self.model.model.parameters()).device)
        except Exception:  # noqa: BLE001
            return self.device_name

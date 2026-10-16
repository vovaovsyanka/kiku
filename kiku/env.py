"""Настройка окружения. Вызывать ДО импорта huggingface_hub / transformers / qwen_tts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def setup_environment(offline: bool) -> None:
    # Все модели Hugging Face лежат внутри проекта: ./models/hf
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if offline:
        # Запрещаем любые сетевые обращения к HF: работаем только с локальным кэшем
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

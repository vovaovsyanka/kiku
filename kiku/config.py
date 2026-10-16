"""Загрузка config.yaml в объект с доступом через точку: cfg.llm.n_ctx"""
from pathlib import Path
from types import SimpleNamespace

import yaml

from .env import ROOT


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _ns(raw)
    cfg.root = ROOT
    return cfg


def resolve_path(p: str) -> Path:
    """~ раскрывается, относительные пути считаются от корня проекта."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else (ROOT / path)

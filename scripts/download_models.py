"""Скачивает модели с Hugging Face в ./models/hf (делается один раз, нужен интернет).

    python scripts/download_models.py                 # всё нужное для голосового режима
    python scripts/download_models.py --only llm      # только языковая модель
    python scripts/download_models.py --with-design   # + модель для придумывания голоса Qwen3-TTS (1.7B)
Голос Silero (tts.engine: silero, ~145 МБ) скачивается с models.silero.ai в ./models/silero. После этого приложение работает полностью офлайн.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kiku.env import setup_environment  # noqa: E402

setup_environment(offline=False)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from kiku.config import load_config  # noqa: E402


def download_silero(cfg) -> None:
    from kiku.voice.engines import silero_model_path, silero_model_url
    path = silero_model_path(cfg)
    if path.exists():
        print(f"[TTS] Silero {cfg.model}: уже есть ({path})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = silero_model_url(cfg)
    print(f"[TTS] Silero {cfg.model} (~145 МБ) <- {url}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["llm", "asr", "tts"], help="скачать только один компонент")
    ap.add_argument("--with-design", action="store_true", help="скачать Qwen3-TTS VoiceDesign (1.7B, ~4-5 ГБ); нужно только для engine: qwen")
    ap.add_argument("--llm-repo", help="скачать другую LLM (например, unsloth/Qwen3-4B-Instruct-2507-GGUF)")
    ap.add_argument("--llm-pattern", default="*Q4_K_M*.gguf", help="маска файла для --llm-repo")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from huggingface_hub import snapshot_download

    print(f"Папка моделей: {os.environ['HF_HOME']}\n")

    if args.llm_repo:
        print(f"[LLM] {args.llm_repo}  ({args.llm_pattern})")
        snapshot_download(repo_id=args.llm_repo, allow_patterns=[args.llm_pattern])
        return
    if args.only in (None, "llm"):
        print(f"[LLM] {cfg.llm.repo_id}  ({cfg.llm.file_pattern}) — это десятки гигабайт, наберитесь терпения")
        snapshot_download(repo_id=cfg.llm.repo_id, allow_patterns=[cfg.llm.file_pattern])

    if args.only in (None, "asr") and cfg.asr.enabled:
        print(f"[ASR] faster-whisper {cfg.asr.model}")
        from faster_whisper.utils import download_model
        download_model(cfg.asr.model)

    if args.only in (None, "tts") and cfg.tts.enabled:
        if cfg.tts.engine == "silero":
            download_silero(cfg.tts.silero)
        else:
            q = cfg.tts.qwen
            repo = q.base_repo if q.mode == "clone" else q.custom_repo
            print(f"[TTS] {repo}")
            snapshot_download(repo_id=repo)
            if args.with_design:
                print(f"[TTS-design] {q.design_repo}")
                snapshot_download(repo_id=q.design_repo)

    print("\nГотово. Дальше: python -m kiku chat --dry-run")


if __name__ == "__main__":
    main()

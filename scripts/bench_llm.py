"""Замер скорости языковой модели на этой машине: разбор промпта и генерация (ток/с).

    python scripts/bench_llm.py                                   # модель из config.yaml
    python scripts/bench_llm.py --repo unsloth/Qwen3-4B-Instruct-2507-GGUF --pattern "*Q4_K_M*.gguf"
    python scripts/bench_llm.py --threads 6 --gpu-layers 0
Модель для --repo сначала скачайте: python scripts/download_models.py --llm-repo ...
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kiku.env import setup_environment  # noqa: E402

setup_environment(offline=True)

from kiku.config import load_config  # noqa: E402
from kiku.llm.llama_cpp_backend import find_gguf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--pattern")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--gpu-layers", type=int)
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()
    cfg = load_config()
    repo = args.repo or cfg.llm.repo_id
    pattern = args.pattern or cfg.llm.file_pattern
    threads = args.threads if args.threads is not None else int(cfg.llm.n_threads)
    gpu = args.gpu_layers if args.gpu_layers is not None else int(cfg.llm.n_gpu_layers)

    from llama_cpp import Llama
    path = find_gguf(repo, pattern)
    print(f"Модель: {repo}\nПотоки: {threads or 'авто'}, слоёв на GPU: {gpu}\nЗагрузка…")
    t0 = time.time()
    llm = Llama(model_path=path, n_ctx=args.ctx, n_gpu_layers=gpu, n_threads=(threads or None), verbose=False)
    print(f"Загружена за {time.time() - t0:.1f} с\n")

    filler = "Кику — домашняя помощница-горничная. Она открывает папки, запускает приложения и помнит задачи. " * 22
    messages = [{"role": "system", "content": filler},
                {"role": "user", "content": "Скажи одним коротким предложением, как дела."}]
    for run in (1, 2):
        t0 = time.time()
        first = None
        n = 0
        for chunk in llm.create_chat_completion(messages=messages, max_tokens=48, temperature=0.3, stream=True):
            if chunk["choices"][0]["delta"].get("content"):
                if first is None:
                    first = time.time() - t0
                n += 1
        total = time.time() - t0
        gen = max(total - (first or 0), 1e-6)
        print(f"Прогон {run}: до первого токена {first:.1f} с (разбор промпта), "
              f"генерация {n - 1} ток. за {gen:.1f} с = {(n - 1) / gen:.1f} ток/с")
    print("\nПрогон 2 быстрее прогона 1 — значит, кэш промпта работает.")


if __name__ == "__main__":
    main()

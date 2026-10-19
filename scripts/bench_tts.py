"""Скорость и голоса озвучки (RTF < 1 — быстрее реального времени).

    python scripts/bench_tts.py --samples          # Silero: замер + wav-файлы всех голосов в data/tts_samples/ (послушайте!)
    python scripts/bench_tts.py --engine qwen      # Qwen3-TTS: варианты внимания, каждый в отдельном процессе

Для Qwen печатается ещё и загрузка GPU по nvidia-smi: низкая загрузка при RTF > 1 означает, что узкое место —
процессор/Python (множество мелких операций), а не видеокарта.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QWEN_VARIANTS = {"bf16-sdpa": ("bfloat16", "sdpa"), "bf16-eager": ("bfloat16", "eager"), "bf16-default": ("bfloat16", None)}
TEXTS = ["Открываю рабочий стол.", "Хорошо, хозяин, сейчас всё сделаю.",
         "Файл заметки создан на рабочем столе, можете его открыть.",
         "Сейчас воскресенье, девятнадцатое октября, четырнадцать часов тридцать минут."]


def bench_silero(save_samples: bool, pitch: str | None, rate: str | None) -> None:
    from kiku.config import load_config, resolve_path
    from kiku.voice.engines import SILERO_SPEAKERS, SileroEngine

    cfg = load_config().tts
    out_dir = resolve_path("data/tts_samples")
    if save_samples:
        out_dir.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
    for speaker in SILERO_SPEAKERS:
        cfg.silero.speaker = speaker
        cfg.silero.pitch, cfg.silero.rate = pitch or cfg.silero.pitch, rate or cfg.silero.rate
        engine = SileroEngine(cfg.silero, pronounce=vars(cfg.pronounce))
        total_dur = total_took = 0.0
        parts = []
        for text in TEXTS:
            t0 = time.time()
            wav, sr = engine.synth(text)
            total_took += time.time() - t0
            total_dur += len(wav) / sr
            parts.append(wav)
        print(f"{speaker:9s} {total_dur:5.1f} с звука за {total_took:4.2f} с  (RTF {total_took / total_dur:.3f})")
        if save_samples:
            import numpy as np
            sf.write(str(out_dir / f"silero_{speaker}_{cfg.silero.pitch}_{cfg.silero.rate}.wav"), np.concatenate(parts), sr)
    if save_samples:
        print(f"\nФайлы лежат в {out_dir}. Понравившийся голос впишите в config.yaml, tts.silero.speaker "
              "(тон и темп — tts.silero.pitch / rate).")


def run_qwen_variant(name: str) -> None:
    from kiku.env import setup_environment
    setup_environment(offline=True)
    from kiku.cuda_init import init_cuda_first
    print(f"[{name}] CUDA: {init_cuda_first()}")

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from _gpu_tools import GpuSampler
    from kiku.config import load_config, resolve_path

    transformers.logging.set_verbosity_error()
    cfg = load_config().tts
    q = cfg.qwen
    dtype_name, attn = QWEN_VARIANTS[name]
    path = snapshot_download(repo_id=q.base_repo, local_files_only=True)
    kwargs = dict(device_map=q.device, dtype=getattr(torch, dtype_name))
    if attn:
        kwargs["attn_implementation"] = attn
    model = Qwen3TTSModel.from_pretrained(path, **kwargs)
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(resolve_path(q.ref_audio)), ref_text=resolve_path(q.ref_text_file).read_text(encoding="utf-8").strip())
    print(f"[{name}] параметры модели на: {next(model.model.parameters()).device}")

    def gen(text):
        with torch.inference_mode():
            return model.generate_voice_clone(text=text, language=cfg.language, voice_clone_prompt=prompt)

    gen(TEXTS[0])                                   # прогрев
    rtfs = []
    with GpuSampler() as gpu:
        for text in TEXTS[:3]:
            t0 = time.time()
            wavs, sr = gen(text)
            took, dur = time.time() - t0, len(wavs[0]) / sr
            rtfs.append(took / dur)
            print(f"[{name}] {dur:.1f} с звука за {took:.1f} с (RTF {took / dur:.2f})")
    print(f"[{name}] СРЕДНИЙ RTF {sum(rtfs) / len(rtfs):.2f}; {gpu.summary()}")


def bench_qwen() -> None:
    print("Каждый вариант в отдельном процессе; первый запуск каждого занимает около минуты.\n")
    summary = []
    for name in QWEN_VARIANTS:
        print(f"=== {name} ===")
        code = subprocess.call([sys.executable, __file__, "--variant", name])
        summary.append((name, "ok" if code == 0 else f"УПАЛ (код {code})"))
        print()
    print("Итог:")
    for name, status in summary:
        print(f"  {name:14s} {status}")
    print("\nfloat16 намеренно не проверяется: на этой видеокарте он уже давал CUDA-ошибку (NaN при сэмплировании).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["silero", "qwen"], default="silero")
    ap.add_argument("--samples", action="store_true", help="сохранить wav каждого голоса Silero")
    ap.add_argument("--pitch", help="тон для проверки: x-low | low | medium | high | x-high")
    ap.add_argument("--rate", help="темп для проверки: x-slow | slow | medium | fast | x-fast")
    ap.add_argument("--variant", help=argparse.SUPPRESS)
    args = ap.parse_args()

    from kiku.env import setup_environment
    setup_environment(offline=True)
    if args.variant:
        run_qwen_variant(args.variant)
    elif args.engine == "qwen":
        bench_qwen()
    else:
        bench_silero(args.samples, args.pitch, args.rate)


if __name__ == "__main__":
    main()

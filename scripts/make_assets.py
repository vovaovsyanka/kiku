"""Нарезает лист с четырьмя позами Кику на отдельные картинки и делает иконку.

    python scripts/make_assets.py                      # берёт kiku/ui/assets/source_sheet.png
    python scripts/make_assets.py путь/к/листу.png

Ожидается лист 2x2 с прозрачным фоном: слева направо, сверху вниз —
  1) стоит и улыбается (idle)  2) рука у уха (listening)  3) задумалась (thinking)  4) говорит (speaking)
Результат: kiku/ui/assets/{idle,listening,thinking,speaking}.png (одинакового размера, выровнены по ногам) и kiku.ico.
Нужны Pillow, numpy, scipy (scipy только для этого скрипта).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ASSETS = Path(__file__).resolve().parent.parent / "kiku" / "ui" / "assets"
NAMES = ["idle", "listening", "thinking", "speaking"]   # TL, TR, BL, BR


def _empty_run_center(profile: np.ndarray, lo: int, hi: int):
    idx = [i for i in range(lo, hi) if profile[i] == 0]
    return (idx[0] + idx[-1]) // 2 if idx else None


def split_owners(alpha: np.ndarray) -> np.ndarray:
    """Каждому непрозрачному пикселю — номер позы 0..3 (TL, TR, BL, BR)."""
    H, W = alpha.shape
    mask = alpha > 40
    cx = _empty_run_center(mask.sum(axis=0), W // 2 - 200, W // 2 + 200) or W // 2
    owner = np.full((H, W), -1, dtype=np.int8)
    for col, (x0, x1) in enumerate([(0, cx), (cx, W)]):
        half = mask[:, x0:x1]
        cy = _empty_run_center(half.sum(axis=1), H // 2 - 150, H // 2 + 150)
        ys, xs = np.where(half)
        if cy is not None:                         # между позами есть пустая строка — режем по ней
            owner[ys, xs + x0] = np.where(ys < cy, col, 2 + col)
            continue
        lab, n = ndi.label(ndi.binary_dilation(half, iterations=6))   # иначе — по связным компонентам
        sizes = ndi.sum(half, lab, range(1, n + 1))
        order = np.argsort(sizes)[::-1][:2] + 1                      # две самые большие компоненты = тела
        centroids = {int(c): ndi.center_of_mass(half, lab, int(c)) for c in range(1, n + 1)}
        top_c, bot_c = sorted(order, key=lambda c: centroids[int(c)][0])
        for c in range(1, n + 1):
            if c in (top_c, bot_c):
                target = col if c == top_c else 2 + col
            else:                                                     # мелкие значки («!», «?») — к ближайшему телу
                dt = abs(centroids[c][0] - centroids[int(top_c)][0])
                db = abs(centroids[c][0] - centroids[int(bot_c)][0])
                target = col if dt <= db else 2 + col
            yy, xx = np.where(lab == c)
            keep = half[yy, xx]
            owner[yy[keep], xx[keep] + x0] = target
    # слабые полупрозрачные пиксели у контура получают владельца ближайшего «твёрдого» пикселя
    idx = ndi.distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
    return owner[idx[0], idx[1]]


def build(sheet: Path) -> None:
    im = Image.open(sheet).convert("RGBA")
    rgba = np.array(im)
    alpha = rgba[:, :, 3]
    owner = split_owners(alpha)

    sprites = []
    for k in range(4):
        part = rgba.copy()
        part[:, :, 3] = np.where((owner == k), alpha, 0)
        ys, xs = np.where(part[:, :, 3] > 24)          # порог отсекает случайные полупрозрачные точки
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = part[y0:y1, x0:x1]
        a = crop[:, :, 3] > 40
        h = crop.shape[0]
        rows = np.where(a[int(h * 0.80):].any(axis=1))[0]
        feet = a[int(h * 0.80):]
        anchor = float(np.where(feet)[1].mean()) if feet.any() else crop.shape[1] / 2   # центр по ногам
        sprites.append((crop, anchor))

    half = int(max(max(a, c.shape[1] - a) for c, a in sprites)) + 4
    height = max(c.shape[0] for c, _ in sprites)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, (crop, anchor) in zip(NAMES, sprites):
        canvas = Image.new("RGBA", (half * 2, height), (0, 0, 0, 0))
        canvas.alpha_composite(Image.fromarray(crop), (int(half - anchor), height - crop.shape[0]))
        canvas.save(ASSETS / f"{name}.png", optimize=True)
        print(f"{name}.png  {canvas.size}")

    # иконка: голова позы «idle»
    idle = Image.open(ASSETS / "idle.png")
    a = np.array(idle)[:, :, 3] > 40
    ys, xs = np.where(a)
    top = ys.min()
    band = a[top + int(0.04 * height): top + int(0.14 * height)]
    cx = int(np.where(band)[1].mean()) if band.any() else idle.width // 2
    side = int(0.40 * height)
    box = (max(cx - side // 2, 0), top, min(cx + side // 2, idle.width), top + side)
    head = idle.crop(box)
    icon = Image.new("RGBA", (max(head.size),) * 2, (0, 0, 0, 0))
    icon.alpha_composite(head, ((icon.width - head.width) // 2, 0))
    icon.save(ASSETS / "kiku.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    icon.save(ASSETS / "kiku_icon.png")
    print("kiku.ico, kiku_icon.png")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ASSETS / "source_sheet.png")

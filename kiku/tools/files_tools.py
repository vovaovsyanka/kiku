"""Наведение порядка в папках: разложить файлы по типам, найти и убрать дубликаты, отменить.

Безопасность: только внутри разрешённых папок, ничего не удаляется и не перезаписывается (файлы лишь переезжают
в подпапки), перед действием — подтверждение хозяина, последнее действие можно отменить (undo_organize).
"""
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from ..config import resolve_path
from .registry import ToolContext, ToolError, ToolRegistry

CATEGORIES = {
    "Изображения": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".heic", ".tiff", ".ico"},
    "Видео": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv"},
    "Музыка": {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"},
    "Документы": {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".rtf", ".odt", ".csv", ".md",
                  ".epub", ".fb2"},
    "Архивы": {".zip", ".rar", ".7z", ".tar", ".gz", ".iso"},
    "Установщики": {".exe", ".msi", ".msix", ".apk"},
}
PARTIAL = {".crdownload", ".part", ".tmp", ".download", ".opdownload"}
MIN_AGE_SEC = 600            # свежие файлы (возможно, ещё скачиваются) не трогаем
DUPLICATES_DIR = "Дубликаты"
MAX_SCAN_FILES = 5000


def category_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    for name, exts in CATEGORIES.items():
        if ext in exts:
            return name
    return None


def plan_organize(folder: Path, now: float | None = None) -> list[tuple[Path, str]]:
    """[(файл, категория)] для файлов прямо в папке (подпапки не трогаем)."""
    now = time.time() if now is None else now
    plan = []
    for item in folder.iterdir():
        if not item.is_file() or item.name.startswith(".") or item.suffix.lower() in PARTIAL:
            continue
        cat = category_of(item)
        try:
            fresh = now - item.stat().st_mtime < MIN_AGE_SEC
        except OSError:
            continue
        if cat and not fresh:
            plan.append((item, cat))
    return plan


def _unique_target(directory: Path, name: str) -> Path:
    target = directory / name
    stem, suffix, n = Path(name).stem, Path(name).suffix, 2
    while target.exists():
        target = directory / f"{stem} ({n}){suffix}"
        n += 1
    return target


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_duplicate_groups(folder: Path) -> list[list[Path]]:
    """Группы одинаковых файлов (по размеру, затем по содержимому) внутри папки, с подпапками. Самый старый файл — первый."""
    by_size: dict[int, list[Path]] = {}
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d != DUPLICATES_DIR and not d.startswith(".")]
        for fn in filenames:
            scanned += 1
            if scanned > MAX_SCAN_FILES:
                break
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > 0:
                by_size.setdefault(size, []).append(p)
    groups = []
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        by_hash: dict[str, list[Path]] = {}
        for p in same_size:
            try:
                by_hash.setdefault(file_hash(p), []).append(p)
            except OSError:
                continue
        groups += [sorted(g, key=lambda x: x.stat().st_mtime) for g in by_hash.values() if len(g) > 1]
    return groups


def register_files_tools(reg: ToolRegistry, ctx: ToolContext) -> None:
    policy = ctx.policy
    undo_path = resolve_path(ctx.cfg.memory.db_path).parent / "organize_undo.json"

    def folder_arg(path: str) -> Path:
        p = policy.resolve(path or "downloads")
        if not p.is_dir():
            raise ToolError(f"папка не найдена: {p.name}")
        return p

    def apply_moves(moves: list[tuple[Path, Path]], created: list[Path]) -> None:
        done = []
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            done.append([str(src), str(dst)])
        undo_path.parent.mkdir(parents=True, exist_ok=True)
        undo_path.write_text(json.dumps({"moves": done, "created": [str(c) for c in created]}, ensure_ascii=False),
                             encoding="utf-8")

    @reg.tool("organize_folder",
              "Разложить файлы папки по подпапкам по типам (Изображения, Документы, Архивы, Видео, Музыка, Установщики). "
              "path по умолчанию downloads. Нужно подтверждение; ничего не удаляется; можно отменить (undo_organize).")
    def organize_folder(path: str = "downloads") -> dict:
        folder = folder_arg(path)
        plan = plan_organize(folder)
        if not plan:
            return {"moved": 0, "note": "раскладывать нечего: подходящих файлов нет или они слишком свежие"}
        counts = Counter(cat for _, cat in plan)
        summary = ", ".join(f"{cat}: {n}" for cat, n in counts.most_common())
        if ctx.dry_run:
            return {"dry_run": True, "would_move": len(plan), "by_category": dict(counts)}
        if not ctx.confirm(f"Разложить {len(plan)} файлов из «{folder.name}» по папкам ({summary})? Потом можно отменить."):
            raise ToolError("хозяин не подтвердил")
        created = [folder / cat for cat in counts if not (folder / cat).exists()]
        apply_moves([(src, _unique_target(folder / cat, src.name)) for src, cat in plan], created)
        return {"moved": len(plan), "by_category": dict(counts)}

    @reg.tool("find_duplicates", "Найти одинаковые файлы (копии) в папке и подпапках; ничего не меняет. path по умолчанию downloads.",
              reads=True)
    def find_duplicates(path: str = "downloads") -> dict:
        groups = find_duplicate_groups(folder_arg(path))
        wasted = sum(p.stat().st_size for g in groups for p in g[1:])
        return {"groups": len(groups), "extra_copies": sum(len(g) - 1 for g in groups),
                "wasted_mb": round(wasted / 2**20),
                "examples": [[p.name for p in g[:3]] for g in groups[:5]]}

    @reg.tool("move_duplicates",
              "Убрать лишние копии: оставить самый старый файл каждой группы, остальные переложить в подпапку «Дубликаты». "
              "Нужно подтверждение; можно отменить (undo_organize).")
    def move_duplicates(path: str = "downloads") -> dict:
        folder = folder_arg(path)
        groups = find_duplicate_groups(folder)
        extras = [p for g in groups for p in g[1:]]
        if not extras:
            return {"moved": 0, "note": "дубликатов нет"}
        if ctx.dry_run:
            return {"dry_run": True, "would_move": len(extras)}
        if not ctx.confirm(f"Переложить {len(extras)} лишних копий из «{folder.name}» в папку «{DUPLICATES_DIR}»?"):
            raise ToolError("хозяин не подтвердил")
        target = folder / DUPLICATES_DIR
        created = [] if target.exists() else [target]
        apply_moves([(p, _unique_target(target, p.name)) for p in extras], created)
        return {"moved": len(extras)}

    @reg.tool("undo_organize", "Отменить последнюю раскладку файлов или перенос дубликатов: вернуть всё на прежние места.")
    def undo_organize() -> dict:
        if not undo_path.exists():
            raise ToolError("отменять нечего: последних переносов нет")
        data = json.loads(undo_path.read_text(encoding="utf-8"))
        moves = [(Path(a), Path(b)) for a, b in data["moves"]]
        for src, dst in moves:                            # безопасность: возвращаем только внутрь разрешённых папок
            policy.check(src)
            policy.check(dst)
        if ctx.dry_run:
            return {"dry_run": True, "would_restore": len(moves)}
        if not ctx.confirm(f"Вернуть {len(moves)} файлов на прежние места?"):
            raise ToolError("хозяин не подтвердил")
        restored = 0
        for src, dst in moves:
            if dst.exists() and not src.exists():
                shutil.move(str(dst), str(src))
                restored += 1
        for folder in map(Path, data.get("created", [])):
            try:
                folder.rmdir()                            # только пустые папки, созданные раскладкой
            except OSError:
                pass
        undo_path.unlink()
        return {"restored": restored, "left_in_place": len(moves) - restored}

"""Инструменты «локального мозга»: факты, заметки, задачи, поиск файлов."""
from typing import Optional

from .registry import ToolContext, ToolError, ToolRegistry


def register_memory_tools(reg: ToolRegistry, ctx: ToolContext) -> None:
    mem = ctx.memory
    policy = ctx.policy

    @reg.tool("remember", "Запомнить. kind='fact' — долговременный факт о хозяине, 'note' — обычная заметка.")
    def remember(text: str, kind: str = "note") -> dict:
        if not text.strip():
            raise ToolError("нечего запоминать")
        if ctx.dry_run:
            return {"dry_run": True}
        return {"saved_id": mem.add_note(text, kind)}

    @reg.tool("recall", "Поиск по заметкам и фактам. query — ключевые слова.", reads=True)
    def recall(query: str, limit: int = 8) -> list:
        return mem.search_notes(query, max(1, min(int(limit), 20)))

    @reg.tool("forget", "Удалить заметку/факт по id (из recall). Нужно подтверждение.")
    def forget(note_id: int) -> str:
        note = mem.get_note(note_id)
        if not note:
            raise ToolError("такой заметки нет")
        if not ctx.confirm(f"Удалить из памяти: «{note['text'][:80]}»?"):
            raise ToolError("хозяин не подтвердил удаление")
        if ctx.dry_run:
            return "[dry-run] удалила бы заметку"
        mem.delete_note(note_id)
        return "заметка удалена"

    @reg.tool("add_task", "Добавить задачу. due — срок ГГГГ-ММ-ДД или ГГГГ-ММ-ДД ЧЧ:ММ (необязательно).")
    def add_task(title: str, due: Optional[str] = None, details: Optional[str] = None) -> dict:
        if not title.strip():
            raise ToolError("пустое название задачи")
        if ctx.dry_run:
            return {"dry_run": True}
        return {"task_id": mem.add_task(title, due, details)}

    @reg.tool("list_tasks", "Список задач. status: open (по умолчанию) | done | all.", reads=True)
    def list_tasks(status: str = "open") -> list:
        status = status if status in ("open", "done", "all") else "open"
        return [{"id": t["id"], "title": t["title"], "due": t["due"], "status": t["status"]}
                for t in mem.list_tasks(status)]

    @reg.tool("complete_task", "Отметить задачу выполненной по id (из list_tasks).")
    def complete_task(task_id: int) -> str:
        if ctx.dry_run:
            return "[dry-run] отметила бы задачу"
        row = mem.complete_task(task_id)
        if not row:
            raise ToolError("задачи с таким id нет")
        return f"задача выполнена: {row['title']}"

    @reg.tool("index_folder",
              "Проиндексировать папку для поиска файлов по названию. path: all (все разрешённые) | desktop | documents | downloads | путь.",
              reads=True)
    def index_folder(path: str = "all") -> dict:
        if ctx.dry_run:
            return {"dry_run": True}
        targets = policy.roots if path.strip().lower() in ("all", "всё", "все", "") else [policy.resolve(path)]
        counts = {}
        for t in targets:
            if t.is_dir():
                counts[str(t)] = mem.index_root(t)
        return {"indexed": counts}

    @reg.tool("find_files", "Найти файлы по названию в индексе. query — слова из названия/папки; ext — расширение.", reads=True)
    def find_files(query: str, ext: Optional[str] = None, limit: int = 6) -> dict:
        found = mem.search_files(query, ext, max(1, min(int(limit), 15)))
        if not found:
            hint = "" if mem.stats()["files"] else " Индекс пуст — предложите хозяину проиндексировать папки (index_folder)."
            return {"found": [], "note": "ничего не найдено." + hint}
        return {"found": [{"path": f["path"], "modified": f["mtime"], "size_kb": round(f["size"] / 1024)}
                          for f in found]}

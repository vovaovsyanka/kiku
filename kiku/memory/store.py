"""Локальная память Кику: SQLite + полнотекстовый поиск FTS5.

Три сущности:
  * notes — факты о хозяине и заметки (kind = 'fact' | 'note')
  * tasks — задачи с необязательным сроком
  * files — индекс имён файлов (содержимое файлов НЕ читается и НЕ хранится)
"""
import os
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

# --- грубый русский "стемминг": срезаем типичные окончания, ищем по префиксу ---
_ENDINGS = re.compile(
    r"(иями|ями|ами|ого|его|ому|ему|ыми|ими|ией|ов|ев|ей|ам|ям|ах|ях|ом|ем|"
    r"ую|юю|ая|яя|ое|ее|ые|ие|ый|ий|ой|ых|их|а|я|у|ю|ы|и|е|о|ь)$"
)
_STOP = {"в", "на", "по", "и", "с", "к", "у", "о", "от", "из", "за", "для", "мой", "моя",
         "мои", "моё", "где", "что", "как", "это", "файл", "лежит", "найди", "покажи"}

SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "$recycle.bin",
             "appdata", "system volume information", ".cache", ".idea", ".vscode"}


def stem(word: str) -> str:
    w = word.lower().replace("ё", "е")
    if len(w) <= 4:
        return w
    s = _ENDINGS.sub("", w)
    return s if len(s) >= 4 else w


def fts_query(text: str, mode: str = "AND") -> str | None:
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP and len(t) > 1]
    if not tokens:
        return None
    return f" {mode} ".join(f'"{stem(t)}"*' for t in tokens)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class MemoryStore:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Закрыть соединение с БД (на Windows открытый файл нельзя удалить)."""
        with self._lock:
            try:
                self.db.close()
            except sqlite3.Error:
                pass

    def _init_schema(self):
        tok = "tokenize='unicode61 remove_diacritics 2'"
        with self._lock, self.db:
            self.db.executescript(f"""
            CREATE TABLE IF NOT EXISTS notes(
                id INTEGER PRIMARY KEY, text TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'note', created_at TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(text, {tok});

            CREATE TABLE IF NOT EXISTS tasks(
                id INTEGER PRIMARY KEY, title TEXT NOT NULL, due TEXT, details TEXT,
                status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, done_at TEXT);

            CREATE TABLE IF NOT EXISTS files(
                id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                ext TEXT, size INTEGER, mtime TEXT, root TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_files_root ON files(root);
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(name, dirs, {tok});
            """)

    # ---------------- заметки и факты ----------------
    def add_note(self, text: str, kind: str = "note") -> int:
        kind = kind if kind in ("fact", "note") else "note"
        with self._lock, self.db:
            cur = self.db.execute("INSERT INTO notes(text, kind, created_at) VALUES(?,?,?)",
                                  (text.strip(), kind, _now()))
            nid = cur.lastrowid
            self.db.execute("INSERT INTO notes_fts(rowid, text) VALUES(?,?)", (nid, text))
        return nid

    def search_notes(self, query: str, limit: int = 8) -> list[dict]:
        with self._lock:
            for mode in ("AND", "OR"):
                q = fts_query(query, mode)
                if not q:
                    return []
                rows = self.db.execute(
                    "SELECT n.id, n.text, n.kind, n.created_at FROM notes_fts "
                    "JOIN notes n ON n.id = notes_fts.rowid WHERE notes_fts MATCH ? "
                    "ORDER BY rank LIMIT ?", (q, limit)).fetchall()
                if rows:
                    return [dict(r) for r in rows]
        return []

    def list_facts(self, limit: int = 30) -> list[str]:
        with self._lock:
            rows = self.db.execute(
                "SELECT text FROM notes WHERE kind='fact' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r["text"] for r in rows][::-1]

    def get_note(self, note_id: int) -> dict | None:
        with self._lock:
            r = self.db.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return dict(r) if r else None

    def delete_note(self, note_id: int) -> bool:
        with self._lock, self.db:
            cur = self.db.execute("DELETE FROM notes WHERE id=?", (note_id,))
            self.db.execute("DELETE FROM notes_fts WHERE rowid=?", (note_id,))
        return cur.rowcount > 0

    # ---------------- задачи ----------------
    def add_task(self, title: str, due: str | None = None, details: str | None = None) -> int:
        with self._lock, self.db:
            cur = self.db.execute(
                "INSERT INTO tasks(title, due, details, created_at) VALUES(?,?,?,?)",
                (title.strip(), due, details, _now()))
        return cur.lastrowid

    def list_tasks(self, status: str = "open", limit: int = 50) -> list[dict]:
        with self._lock:
            if status == "all":
                rows = self.db.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY (due IS NULL), due, id LIMIT ?",
                    (status, limit)).fetchall()
        return [dict(r) for r in rows]

    def complete_task(self, task_id: int) -> dict | None:
        with self._lock, self.db:
            row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return None
            self.db.execute("UPDATE tasks SET status='done', done_at=? WHERE id=?", (_now(), task_id))
        return dict(row)

    # ---------------- индекс файлов ----------------
    def index_root(self, root: Path, max_files: int = 300_000) -> int:
        """Сканирует папку и заменяет её записи в индексе. Читает только имена и метаданные."""
        root = Path(root)
        rows = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            rel_dirs = " ".join(Path(dirpath).relative_to(root).parts)
            for fn in filenames:
                if len(rows) >= max_files:
                    break
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                rows.append((full, fn, Path(fn).suffix.lower(), st.st_size,
                             datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                             str(root), rel_dirs))
            if len(rows) >= max_files:
                break
        with self._lock, self.db:
            old = [r["id"] for r in self.db.execute("SELECT id FROM files WHERE root=?", (str(root),))]
            self.db.executemany("DELETE FROM files_fts WHERE rowid=?", [(i,) for i in old])
            self.db.execute("DELETE FROM files WHERE root=?", (str(root),))
            for path, name, ext, size, mtime, rt, dirs in rows:
                cur = self.db.execute(
                    "INSERT OR REPLACE INTO files(path,name,ext,size,mtime,root) VALUES(?,?,?,?,?,?)",
                    (path, name, ext, size, mtime, rt))
                self.db.execute("INSERT INTO files_fts(rowid, name, dirs) VALUES(?,?,?)",
                                (cur.lastrowid, name, dirs))
        return len(rows)

    def search_files(self, query: str, ext: str | None = None, limit: int = 8) -> list[dict]:
        with self._lock:
            for mode in ("AND", "OR"):
                q = fts_query(query, mode)
                if not q:
                    return []
                sql = ("SELECT f.path, f.name, f.size, f.mtime FROM files_fts "
                       "JOIN files f ON f.id = files_fts.rowid WHERE files_fts MATCH ?")
                params: list = [q]
                if ext:
                    sql += " AND f.ext = ?"
                    params.append(ext.lower() if ext.startswith(".") else "." + ext.lower())
                sql += " ORDER BY bm25(files_fts, 5.0, 1.0) LIMIT ?"
                params.append(limit * 3)
                rows = [dict(r) for r in self.db.execute(sql, params).fetchall()]
                rows = [r for r in rows if os.path.exists(r["path"])][:limit]
                if rows:
                    return rows
        return []

    def stats(self) -> dict:
        with self._lock:
            c = lambda t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            return {"notes": c("notes"), "tasks": c("tasks"), "files": c("files")}

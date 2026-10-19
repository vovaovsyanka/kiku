"""Врач для ПК: загрузка процессора и памяти, место на дисках, батарея, «что тормозит».

Читает данные через psutil. Закрыть программу можно только по имени, с подтверждением, и не из списка системных.
"""
import getpass
import os
import time

from ..voice.ru_text import plural
from .registry import ToolContext, ToolError, ToolRegistry

# Эти процессы Кику не закрывает никогда: без них Windows или сама Кику перестанет работать.
PROTECTED = {
    "system", "system idle process", "registry", "smss", "csrss", "wininit", "winlogon", "services", "lsass", "svchost",
    "dwm", "explorer", "fontdrvhost", "sihost", "taskhostw", "ctfmon", "conhost", "searchhost", "startmenuexperiencehost",
    "shellexperiencehost", "runtimebroker", "audiodg", "spoolsv", "python", "pythonw", "kiku", "claude", "powershell", "cmd",
    "windowsterminal", "wt",
}


def _short_name(name: str) -> str:
    n = name.lower()
    return n[:-4] if n.endswith(".exe") else n


def _gb(n: float) -> float:
    return round(n / 2**30, 1)


def processes_by(psutil, by: str, limit: int) -> list[dict]:
    """Процессы, сгруппированные по имени (у Chrome их десятки): {name, memory_mb, cpu}."""
    groups: dict[str, dict] = {}
    procs = list(psutil.process_iter(["name", "memory_info", "cpu_percent"]))
    for p in procs:
        info = p.info
        mi = info.get("memory_info")
        if not info.get("name") or mi is None or _short_name(info["name"]) in ("system idle process", "idle"):
            continue
        g = groups.setdefault(_short_name(info["name"]), {"name": _short_name(info["name"]), "memory_mb": 0, "cpu": 0.0})
        g["memory_mb"] += mi.rss / 2**20
        g["cpu"] += info.get("cpu_percent") or 0.0
    key = "cpu" if by == "cpu" else "memory_mb"
    top = sorted(groups.values(), key=lambda g: g[key], reverse=True)[:limit]
    return [{"name": g["name"], "memory_mb": round(g["memory_mb"]), "cpu": round(g["cpu"] / (psutil.cpu_count() or 1), 1)}
            for g in top]


def register_pc_tools(reg: ToolRegistry, ctx: ToolContext) -> None:
    try:
        import psutil
    except ImportError:
        psutil = None

    def need_psutil():
        if psutil is None:
            raise ToolError("для диагностики ПК нужен пакет psutil: pip install psutil")
        return psutil

    @reg.tool("system_status", "Состояние компьютера: загрузка процессора, память, свободное место на дисках, батарея, время работы.",
              reads=True)
    def system_status() -> dict:
        ps = need_psutil()
        vm = ps.virtual_memory()
        disks = []
        for part in ps.disk_partitions(all=False):
            if "cdrom" in part.opts or not part.fstype:
                continue
            try:
                u = ps.disk_usage(part.mountpoint)
            except OSError:
                continue
            disks.append({"disk": part.mountpoint.rstrip("\\/"), "free_gb": _gb(u.free), "total_gb": _gb(u.total)})
        out = {"cpu_percent": ps.cpu_percent(interval=0.4), "ram_percent": round(vm.percent),
               "ram_used_gb": _gb(vm.used), "ram_total_gb": _gb(vm.total), "disks": disks,
               "uptime_hours": round((time.time() - ps.boot_time()) / 3600, 1)}
        bat = ps.sensors_battery() if hasattr(ps, "sensors_battery") else None
        if bat is not None:
            out["battery_percent"], out["plugged_in"] = round(bat.percent), bool(bat.power_plugged)
        return out

    @reg.tool("top_processes", "Какие программы больше всего грузят компьютер. by: memory (по умолчанию) | cpu; limit — сколько показать.",
              reads=True)
    def top_processes(by: str = "memory", limit: int = 5) -> list:
        ps = need_psutil()
        list(ps.process_iter(["cpu_percent"]))      # первый замер только запоминает время; через полсекунды цифры настоящие
        time.sleep(0.5)
        return processes_by(ps, "cpu" if by == "cpu" else "memory", max(1, min(int(limit), 10)))

    @reg.tool("close_process", "Закрыть программу по имени процесса (из top_processes), например chrome. Нужно подтверждение.")
    def close_process(name: str) -> str:
        ps = need_psutil()
        short = _short_name(name.strip())
        if not short:
            raise ToolError("не указано, что закрыть")
        if short in PROTECTED:
            raise ToolError(f"«{short}» закрывать нельзя: это системная программа или сама Кику")
        me = getpass.getuser().lower()
        victims = []
        for p in ps.process_iter(["name", "username"]):
            if _short_name(p.info.get("name") or "") == short and me in (p.info.get("username") or "").lower() and p.pid != os.getpid():
                victims.append(p)
        if not victims:
            raise ToolError(f"программа «{short}» не запущена")
        if not ctx.confirm(f"Закрыть «{short}»? Несохранённые данные могут пропасть."):
            raise ToolError("хозяин не подтвердил закрытие")
        if ctx.dry_run:
            return f"[dry-run] закрыла бы «{short}» ({len(victims)} процессов)"
        for p in victims:
            try:
                p.terminate()
            except ps.Error:
                pass
        ps.wait_procs(victims, timeout=3)
        return f"закрыто: {short}"


# ---------- человеческие формулировки (для быстрых ответов без LLM) ----------
def _percent(n) -> str:
    n = int(round(n))
    return f"{n} {plural(n, 'процент', 'процента', 'процентов')}"


def _program(name: str) -> str:
    return {"chrome": "Хром", "msedge": "Эдж", "firefox": "Фаерфокс", "code": "Ви Эс Код", "discord": "Дискорд",
            "telegram": "Телеграм", "steam": "Стим", "cs2": "Контр Страйк"}.get(name, name)


def _size(mb: float) -> str:
    gb = round(mb / 1024)
    if mb >= 1536:
        return f"{gb} {plural(gb, 'гигабайт', 'гигабайта', 'гигабайт')}"
    mb = max(10, round(mb / 10) * 10)
    return f"{mb} {plural(mb, 'мегабайт', 'мегабайта', 'мегабайт')}"


def describe_load(status: dict, top_mem: list, top_cpu: list) -> str:
    """«Процессор занят на 35 процентов, память на 72. Больше всего памяти берут: Хром 2 гигабайта, …»"""
    parts = [f"Процессор занят на {_percent(status['cpu_percent'])}, память на {_percent(status['ram_percent'])}."]
    if top_mem:
        mem = ", ".join(f"{_program(p['name'])} {_size(p['memory_mb'])}" for p in top_mem[:3])
        parts.append(f"Больше всего памяти берут: {mem}.")
    hot = [p for p in top_cpu[:1] if p["cpu"] >= 15]
    if hot:
        parts.append(f"Процессор сильнее всего грузит {_program(hot[0]['name'])}.")
    return " ".join(parts)


def describe_disks(status: dict) -> str:
    def gb(n):
        n = round(n)
        return f"{n} {plural(n, 'гигабайт', 'гигабайта', 'гигабайт')}"
    return ". ".join(f"На диске {d['disk'].rstrip(':')} свободно {gb(d['free_gb'])} из {gb(d['total_gb'])}"
                     for d in status["disks"][:3]) + "."


def describe_battery(status: dict) -> str:
    if "battery_percent" not in status:
        return "У этого компьютера нет батареи."
    plug = "и он на зарядке" if status["plugged_in"] else "от батареи"
    return f"Заряд {_percent(status['battery_percent'])}, {plug}."

"""Сцены («режимы»): одна фраза запускает цепочку действий, описанную в config.yaml → scenes.

Шаги — это обычные инструменты Кику (launch_app, volume, window_action…), поэтому все проверки безопасности
(белый список приложений, подтверждения, dry-run) действуют и внутри сцены.
"""
import logging

from .registry import ToolContext, ToolError, ToolRegistry

log = logging.getLogger("kiku.scenes")


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("ё", "е").split())


def scene_table(cfg) -> dict:
    """{нормализованное название или синоним: сцена}."""
    table = {}
    for sc in (getattr(cfg, "scenes", None) or []):
        for key in [sc.id, sc.title, *getattr(sc, "aliases", [])]:
            table[_norm(key)] = sc
    return table


def find_scene(cfg, name: str):
    return scene_table(cfg).get(_norm(name))


def run_scene(reg: ToolRegistry, scene) -> tuple[bool, list[str]]:
    """Выполняет шаги сцены по порядку. Возвращает (всё ли удалось, список ошибок)."""
    errors = []
    for step in scene.steps:
        args = vars(step.args) if getattr(step, "args", None) is not None else {}
        res = reg.call(step.tool, args)
        if not res.get("ok"):
            errors.append(f"{step.tool}: {res.get('error')}")
    return not errors, errors


def validate_scenes(reg: ToolRegistry, cfg) -> list[str]:
    """Проблемы в конфиге сцен (неизвестные инструменты) — для лога при старте."""
    problems = []
    for sc in (getattr(cfg, "scenes", None) or []):
        for step in sc.steps:
            if step.tool not in reg.tools:
                problems.append(f"сцена «{sc.id}»: неизвестный инструмент «{step.tool}»")
    return problems


def register_scene_tools(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool("list_scenes", "Какие режимы (сцены) умеет включать Кику.", reads=True)
    def list_scenes() -> list:
        seen, out = set(), []
        for sc in scene_table(ctx.cfg).values():
            if sc.id not in seen:
                seen.add(sc.id)
                out.append({"id": sc.id, "title": sc.title, "phrases": list(getattr(sc, "aliases", []))[:3]})
        return out

    @reg.tool("run_scene", "Включить режим (сцену): «игровой режим», «рабочий режим», «спокойной ночи». name — название режима.")
    def run_scene_tool(name: str) -> str:
        scene = find_scene(ctx.cfg, name)
        if scene is None:
            raise ToolError(f"режима «{name}» нет в config.yaml → scenes")
        ok, errors = run_scene(reg, scene)
        if not ok:
            raise ToolError("режим включён не полностью: " + "; ".join(errors))
        return f"режим «{scene.title}» включён"

"""Сборка приложения: память, инструменты, LLM, агент."""
import logging
from dataclasses import dataclass

from .agent import Agent, Router
from .audit import AuditLog
from .config import resolve_path
from .memory import MemoryStore
from .proactive import briefing_text
from .tools import (PathPolicy, ToolContext, ToolRegistry, register_clipboard_tools, register_files_tools,
                    register_memory_tools, register_net_tools, register_pc_tools, register_reminder_tools,
                    register_scene_tools, register_system_tools, register_window_tools)
from .tools.scene_tools import validate_scenes

log = logging.getLogger("kiku.app")


@dataclass
class App:
    cfg: object
    agent: Agent
    ctx: ToolContext
    registry: ToolRegistry
    memory: MemoryStore


def build_app(cfg, llm=None, dry_run: bool | None = None, win_backend=None, clipboard=None) -> App:
    """llm, win_backend и clipboard можно передать готовые (для тестов)."""
    audit = AuditLog(resolve_path(cfg.memory.audit_path))
    memory = MemoryStore(resolve_path(cfg.memory.db_path))
    policy = PathPolicy(cfg)
    ctx = ToolContext(cfg=cfg, memory=memory, policy=policy, audit=audit,
                      dry_run=bool(cfg.security.dry_run if dry_run is None else dry_run))
    registry = ToolRegistry(audit)
    register_system_tools(registry, ctx)
    win_backend = register_window_tools(registry, ctx, win_backend)
    register_memory_tools(registry, ctx)
    register_reminder_tools(registry, ctx)
    register_scene_tools(registry, ctx)
    register_pc_tools(registry, ctx)
    register_clipboard_tools(registry, ctx, win_backend, clipboard)
    register_files_tools(registry, ctx)
    register_net_tools(registry, ctx)
    for problem in validate_scenes(registry, cfg):
        log.warning(problem)

    router_cfg = getattr(cfg, "router", None)
    router = None
    if router_cfg is None or getattr(router_cfg, "enabled", True):
        router = Router(registry, cfg, briefing_fn=lambda: briefing_text(cfg, memory, registry))

    if llm is None:
        from .llm import build_llm
        llm = build_llm(cfg)
    agent = Agent(llm, registry, memory, cfg, router=router)
    return App(cfg=cfg, agent=agent, ctx=ctx, registry=registry, memory=memory)

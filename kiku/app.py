"""Сборка приложения: память, инструменты, LLM, агент."""
import logging
from dataclasses import dataclass

from .agent import Agent, Router
from .audit import AuditLog
from .config import resolve_path
from .memory import MemoryStore
from .tools import (PathPolicy, ToolContext, ToolRegistry, register_memory_tools,
                    register_system_tools, register_window_tools)

log = logging.getLogger("kiku.app")


@dataclass
class App:
    cfg: object
    agent: Agent
    ctx: ToolContext
    registry: ToolRegistry
    memory: MemoryStore


def build_app(cfg, llm=None, dry_run: bool | None = None, win_backend=None) -> App:
    """llm и win_backend можно передать готовые (для тестов)."""
    audit = AuditLog(resolve_path(cfg.memory.audit_path))
    memory = MemoryStore(resolve_path(cfg.memory.db_path))
    policy = PathPolicy(cfg)
    ctx = ToolContext(cfg=cfg, memory=memory, policy=policy, audit=audit,
                      dry_run=bool(cfg.security.dry_run if dry_run is None else dry_run))
    registry = ToolRegistry(audit)
    register_system_tools(registry, ctx)
    register_window_tools(registry, ctx, win_backend)
    register_memory_tools(registry, ctx)

    router_cfg = getattr(cfg, "router", None)
    router = Router(registry, cfg) if (router_cfg is None or getattr(router_cfg, "enabled", True)) else None

    if llm is None:
        from .llm import build_llm
        llm = build_llm(cfg)
    agent = Agent(llm, registry, memory, cfg, router=router)
    return App(cfg=cfg, agent=agent, ctx=ctx, registry=registry, memory=memory)

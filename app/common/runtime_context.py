"""会话级运行时上下文（Sprint 3 P0 修复）。

让工具函数能拿到当前会话的 ``thread_id``，**而不依赖 LLM 在工具调用参数里
正确传入**。LangGraph 的 ``configurable`` 对 LLM 并不可见，让 LLM "从
configurable 里取 thread_id" 在实践中不可靠；改为：

1. FastAPI 入口 ``causal_chat`` 在调用 ``agent.stream`` 之前，把当前
   ``thread_id`` 写入 :data:`CURRENT_THREAD_ID`（``contextvars.ContextVar``）；
2. ``summarize_uploaded_dataset`` / ``recommend_causal_discovery_algorithms``
   等需要会话上下文的工具，从该 ContextVar 读，不再让 LLM 传 ``thread_id``。

``ContextVar`` 在 ``asyncio`` 任务中是按任务隔离的，多并发会话不会相互污染。
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

CURRENT_THREAD_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "causal_agent_current_thread_id",
    default=None,
)


def get_current_thread_id() -> Optional[str]:
    """读取当前会话 thread_id；未设置时返回 None。"""
    return CURRENT_THREAD_ID.get()


@contextmanager
def bind_thread_id(thread_id: Optional[str]) -> Iterator[None]:
    """用 ``with`` 暂时绑定当前会话 thread_id：

    >>> with bind_thread_id("causal-abc"):
    ...     assert get_current_thread_id() == "causal-abc"

    退出 ``with`` 块后自动恢复原值。供测试 / 工具内部临时切换使用。
    """
    token = CURRENT_THREAD_ID.set(thread_id)
    try:
        yield
    finally:
        CURRENT_THREAD_ID.reset(token)

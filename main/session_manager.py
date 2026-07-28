"""会话管理 - 跟踪活跃时间、过期清理、Agent 生命周期"""

import asyncio
import os
import time
from typing import Callable


SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
SESSION_CLEANUP_INTERVAL = int(os.environ.get("SESSION_CLEANUP_INTERVAL", "300"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))


class SessionManager:
    """管理 Agent 实例与最后访问时间，支持惰性 + 定期清理"""

    def __init__(self, create_agent: Callable):
        self._agents: dict = {}
        self._last_access: dict[str, float] = {}
        self._create_agent = create_agent

    def get_agent(self, session_id: str):
        sid = session_id or "default"
        if sid not in self._agents:
            if len(self._agents) >= MAX_SESSIONS:
                self._evict_oldest()
            self._agents[sid] = self._create_agent()
        self._last_access[sid] = time.monotonic()
        return self._agents[sid]

    def touch(self, session_id: str) -> None:
        sid = session_id or "default"
        if sid in self._agents:
            self._last_access[sid] = time.monotonic()

    def remove(self, session_id: str) -> None:
        sid = session_id or "default"
        self._agents.pop(sid, None)
        self._last_access.pop(sid, None)

    def cleanup_expired(self) -> int:
        """清理超过 TTL 未活动的会话，返回清理数量"""
        now = time.monotonic()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.remove(sid)
        return len(expired)

    def _evict_oldest(self) -> None:
        """会话数达上限时，淘汰最久未活动的会话"""
        if not self._last_access:
            return
        oldest = min(self._last_access, key=self._last_access.get)
        self.remove(oldest)

    @property
    def active_count(self) -> int:
        return len(self._agents)

    def clear(self) -> None:
        for sid in list(self._agents.keys()):
            self.remove(sid)


async def run_cleanup_loop(manager: SessionManager, interval: int = SESSION_CLEANUP_INTERVAL):
    """后台定期清理过期会话，不阻塞请求热路径"""
    while True:
        await asyncio.sleep(interval)
        removed = manager.cleanup_expired()
        if removed:
            print(f"[session] 已清理 {removed} 个过期会话，当前活跃: {manager.active_count}")

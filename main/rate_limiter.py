"""轻量级内存限流器 - O(1) 固定窗口，无外部依赖"""

import os
import time


class FixedWindowRateLimiter:
    """固定窗口计数限流，单次检查为 O(1)"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, tuple[int, int]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        """检查是否允许请求，返回 (allowed, retry_after_seconds)"""
        now = int(time.time())
        window_start, count = self._windows.get(key, (now, 0))

        if now - window_start >= self.window_seconds:
            window_start, count = now, 0

        if count >= self.max_requests:
            retry_after = max(1, window_start + self.window_seconds - now)
            return False, retry_after

        self._windows[key] = (window_start, count + 1)
        return True, 0

    def cleanup_stale(self, max_age_seconds: int = 3600) -> int:
        """清理过期窗口记录，防止内存无限增长"""
        now = int(time.time())
        stale = [
            key for key, (window_start, _) in self._windows.items()
            if now - window_start > max_age_seconds
        ]
        for key in stale:
            del self._windows[key]
        return len(stale)

    @property
    def tracked_keys(self) -> int:
        return len(self._windows)


# 环境变量配置（可通过 .env 调整）
CHAT_RATE_PER_SESSION = int(os.environ.get("CHAT_RATE_PER_SESSION", "30"))
CHAT_RATE_PER_IP = int(os.environ.get("CHAT_RATE_PER_IP", "60"))
CHAT_RATE_WINDOW = int(os.environ.get("CHAT_RATE_WINDOW", "60"))

session_chat_limiter = FixedWindowRateLimiter(CHAT_RATE_PER_SESSION, CHAT_RATE_WINDOW)
ip_chat_limiter = FixedWindowRateLimiter(CHAT_RATE_PER_IP, CHAT_RATE_WINDOW)

"""限流器 - 支持 Redis（多 worker 共享计数）与内存（单 worker 回退）两种后端

并发模型说明（高并发改造后）：
  - 配置 REDIS_URL 时，使用 Redis 原子 INCR+EXPIRE 实现固定窗口限流，
    多 worker / 多进程看到一致计数，杜绝单 worker 下 IP 限流被 worker 数倍绕过。
  - 未配置 REDIS_URL 时回退到进程内字典（仅适用单 worker 开发）。
  - Redis 故障时【不放过也不误杀】：单次检查抛异常由 API 层返回 503，
    避免限流被静默绕过导致 LLM 被打爆。
  - 窗口键按时间桶设计（window_start 对齐到窗口边界），同一桶内 INCR 原子递增，
    EXPIRE 幂等设置，避免竞态。
"""

import os
import time
import logging

logger = logging.getLogger("rate_limiter")


# 环境变量配置（可通过 .env 调整）
CHAT_RATE_PER_SESSION = int(os.environ.get("CHAT_RATE_PER_SESSION", "30"))
CHAT_RATE_PER_IP = int(os.environ.get("CHAT_RATE_PER_IP", "60"))
CHAT_RATE_WINDOW = int(os.environ.get("CHAT_RATE_WINDOW", "60"))


class FixedWindowRateLimiter:
    """固定窗口计数限流，单次检查为 O(1)（进程内，仅单 worker）"""

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


class RedisFixedWindowRateLimiter:
    """Redis 固定窗口限流（多 worker 共享计数）。

    采用时间桶键：key = prefix:bucket_start:identity
    bucket_start = now - (now % window)，确保同一窗口内所有请求命中同一计数器。
    INCR 原子递增；EXPIRE 幂等设置（覆盖也无害），TTL 设为窗口的 2 倍兜底。
    """

    def __init__(self, redis_client, max_requests: int, window_seconds: int, prefix: str):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.prefix = prefix

    def allow(self, key: str) -> tuple[bool, int]:
        now = int(time.time())
        bucket = now - (now % self.window_seconds)
        rkey = f"{self.prefix}:{bucket}:{key}"
        # pipeline 以 MULTI 保证 INCR+EXPIRE 原子可见
        pipe = self.redis.pipeline(transaction=True)
        pipe.incr(rkey)
        pipe.expire(rkey, self.window_seconds * 2)
        count, _ = pipe.execute()

        if count > self.max_requests:
            ttl = self.redis.ttl(rkey)
            return False, max(1, ttl)
        return True, 0

    def cleanup_stale(self, max_age_seconds: int = 3600) -> int:
        """Redis 模式下由 TTL 自动回收，no-op。"""
        return 0

    @property
    def tracked_keys(self) -> int:
        return 0


# ======================== 全局限流器（在 lifespan 中按 Redis 配置初始化） ========================
# 维持原模块级名称以兼容 api_server 的 import，但实际实例在 init_limiters() 后赋值。
ip_chat_limiter = None
session_chat_limiter = None


def init_limiters(redis_client=None) -> None:
    """在应用启动时初始化限流器。

    - redis_client 非 None：使用 Redis 共享限流（多 worker 推荐）
    - redis_client 为 None：回退到进程内存限流（仅单 worker 开发）
    必须在 lifespan 启动阶段调用一次。
    """
    global ip_chat_limiter, session_chat_limiter
    if redis_client is not None:
        session_chat_limiter = RedisFixedWindowRateLimiter(
            redis_client, CHAT_RATE_PER_SESSION, CHAT_RATE_WINDOW, "menu:rl:ses"
        )
        ip_chat_limiter = RedisFixedWindowRateLimiter(
            redis_client, CHAT_RATE_PER_IP, CHAT_RATE_WINDOW, "menu:rl:ip"
        )
        logger.info("限流器已启用 Redis 共享计数（多 worker 一致）")
    else:
        session_chat_limiter = FixedWindowRateLimiter(CHAT_RATE_PER_SESSION, CHAT_RATE_WINDOW)
        ip_chat_limiter = FixedWindowRateLimiter(CHAT_RATE_PER_IP, CHAT_RATE_WINDOW)
        logger.info("限流器使用进程内存计数（仅适用单 worker）")

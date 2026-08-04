"""会话管理 - 会话历史外部化到 Redis，支持多 worker 共享（防串号）

设计要点（高并发改造后）：
  - 历史（HumanMessage/AIMessage）序列化为 JSON 存入 Redis，键为 session_id，
    TTL=SESSION_TTL_SECONDS。每个 session_id 独立 Redis key，物理隔离，杜绝串号。
  - 单个 worker 持有一个共享无状态 OrderingAgent（graph 只读、可并发），
    所有会话复用该 agent，避免每会话重建 graph。
  - 未配置 REDIS_URL 时自动回退到进程内存字典（仅适用单 worker 开发模式）。
  - Redis 故障时单条请求降级：抛出异常由 API 层返回 503，绝不回退到内存字典
    （否则多 worker 下会出现部分 worker 用内存、部分用 Redis，导致历史错乱）。
  - MAX_SESSIONS 作为 Redis 侧会话计数上限（粗略保护），由健康检查暴露。
"""

import asyncio
import json
import os
import secrets
import threading
import time
import uuid
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage


SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
SESSION_CLEANUP_INTERVAL = int(os.environ.get("SESSION_CLEANUP_INTERVAL", "300"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))

# 单会话并发锁超时（秒）：>= LLM_REQUEST_TIMEOUT + 工具执行余量。
# 持有锁的请求崩溃时，TTL 到期自动释放，避免会话被永久锁死。
SESSION_LOCK_TIMEOUT = int(os.environ.get("SESSION_LOCK_TIMEOUT", "45"))

# session_token 长度（字节），hex 编码后 64 字符
SESSION_TOKEN_BYTES = 32


class SessionAuthError(Exception):
    """会话鉴权失败（session_token 缺失或不匹配）。

    由 API 层映射为 401，区别于 SessionBusyError（429）。
    """


class SessionBusyError(Exception):
    """同一会话已有请求正在处理中。

    SessionManager.chat 通过每会话锁串行化同一 session_id 的并发请求，
    防止 load-modify-save 竞态导致历史覆写丢失（多 tab / 双击 / 重试场景）。
    由 API 层映射为 429。
    """


# Redis 锁释放 Lua 脚本：仅当 token 匹配时才删除，防止误删他人持有的锁
_RELEASE_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# Redis key 前缀
_SESSION_PREFIX = "menu:session:"
# 会话索引集合（用于统计活跃会话数，非精确限流）
_SESSION_INDEX_KEY = "menu:session:index"


def _serialize_history(messages: list) -> str:
    """将 LangChain 消息列表序列化为 JSON。

    仅持久化 HumanMessage / AIMessage 的文本内容（与原实现一致：ToolMessage 不入历史）。
    """
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"r": "h", "c": m.content})
        elif isinstance(m, AIMessage):
            out.append({"r": "a", "c": m.content})
    return json.dumps(out, ensure_ascii=False)


def _deserialize_history(raw: str) -> list:
    """从 JSON 还原消息列表。"""
    data = json.loads(raw)
    msgs = []
    for item in data:
        if item.get("r") == "h":
            msgs.append(HumanMessage(content=item["c"]))
        else:
            msgs.append(AIMessage(content=item["c"]))
    return msgs


class SessionManager:
    """管理会话历史与共享无状态 Agent。

    Args:
        agent: 共享无状态 OrderingAgent 实例
        redis_client: 已创建的 Redis 客户端；为 None 时回退到进程内存（仅单 worker）
    """

    def __init__(self, agent, redis_client=None):
        self.agent = agent
        self.redis = redis_client
        # 内存回退模式专用
        self._mem: dict[str, list] = {}
        self._mem_access: dict[str, float] = {}
        self._mem_tokens: dict[str, str] = {}
        # 内存模式下的每会话锁（Redis 模式使用 Redis 分布式锁）
        self._mem_locks: dict[str, threading.Lock] = {}
        self._mem_locks_guard = threading.Lock()

    # ---------- session_token 签发/校验/销毁 ----------
    def _token_key(self, session_id: str) -> str:
        return f"menu:session:token:{session_id}"

    def issue_token(self, session_id: str) -> str:
        """为会话签发 token 并持久化。会话首次创建时调用。

        若该会话已有 token，覆盖重签（用于 session 被劫持后强制重置场景）。
        """
        token = secrets.token_hex(SESSION_TOKEN_BYTES)
        if self.redis is not None:
            self.redis.setex(self._token_key(session_id), SESSION_TTL_SECONDS, token)
        else:
            self._mem_tokens[session_id] = token
        return token

    def verify_token(self, session_id: str, token: str) -> bool:
        """校验 session_token 是否匹配（常数时间比较，防时序攻击）。

        Returns:
            True 表示 token 与已签发凭证匹配；False 表示凭证缺失/会话不存在/不匹配。
            新会话的签发统一走 issue_token（由 _verify_session_access 在 session_id 为空时调用），
            此处不再对 stored is None 放行——否则攻击者可伪造任意 session_id 绕过 cart/reset 鉴权。
        """
        if not token:
            return False
        if self.redis is not None:
            stored = self.redis.get(self._token_key(session_id))
        else:
            stored = self._mem_tokens.get(session_id)
        # 会话无已签发 token：视为无效（防止伪造 session_id 绕过对象级鉴权）
        if stored is None:
            return False
        if not isinstance(stored, str):
            stored = stored.decode("utf-8") if isinstance(stored, (bytes, bytearray)) else str(stored)
        return secrets.compare_digest(stored, token)

    def _delete_token(self, session_id: str) -> None:
        if self.redis is not None:
            self.redis.delete(self._token_key(session_id))
        else:
            self._mem_tokens.pop(session_id, None)

    # ---------- 核心对话 ----------
    def chat(self, session_id: str, user_input: str, membership_level: str = "") -> str:
        """加载历史 -> 调用 agent -> 回写历史。

        通过每会话锁串行化同一 session_id 的并发请求，防止 load-modify-save
        竞态导致历史覆写丢失（多 tab / 双击 / 重试场景下，两个并发请求会
        各自读到旧历史并相互覆写，丢失一整轮对话）。
        """
        lock_token = self._acquire_lock(session_id)
        try:
            history = self._load_history(session_id)
            response, new_msgs = self.agent.chat(user_input, history, membership_level)
            new_history = (history + new_msgs)[-self.agent.MAX_HISTORY:]
            self._save_history(session_id, new_history)
            return response
        finally:
            self._release_lock(session_id, lock_token)

    def reset(self, session_id: str) -> None:
        """清空指定会话历史与 token。"""
        self._delete_history(session_id)
        self._delete_token(session_id)

    # ---------- 会话并发锁 ----------
    def _lock_key(self, session_id: str) -> str:
        return f"menu:session:lock:{session_id}"

    def _acquire_lock(self, session_id: str) -> Optional[str]:
        """获取会话锁（非阻塞）。成功返回 token（Redis 模式）或 None（内存模式）；
        失败抛 SessionBusyError。

        - Redis 模式：SET NX EX 实现跨 worker 分布式锁，TTL 兜底防死锁。
        - 内存模式：threading.Lock 非阻塞获取（仅单 worker 开发）。
        """
        if self.redis is not None:
            token = str(uuid.uuid4())
            ok = self.redis.set(
                self._lock_key(session_id), token, nx=True, ex=SESSION_LOCK_TIMEOUT
            )
            if not ok:
                raise SessionBusyError(session_id)
            return token
        # 内存模式：非阻塞获取，已持有则立即拒绝
        with self._mem_locks_guard:
            lock = self._mem_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._mem_locks[session_id] = lock
        if not lock.acquire(blocking=False):
            raise SessionBusyError(session_id)
        return None

    def _release_lock(self, session_id: str, token: Optional[str]) -> None:
        """释放会话锁。"""
        if self.redis is not None:
            if token:
                try:
                    self.redis.eval(
                        _RELEASE_LOCK_SCRIPT, 1, self._lock_key(session_id), token
                    )
                except Exception:
                    pass
            return
        # 内存模式
        with self._mem_locks_guard:
            lock = self._mem_locks.get(session_id)
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                pass  # 锁已被释放（异常恢复场景），忽略

    # ---------- 历史存取 ----------
    def _key(self, session_id: str) -> str:
        return f"{_SESSION_PREFIX}{session_id}"

    def _load_history(self, session_id: str) -> list:
        if self.redis is not None:
            raw = self.redis.get(self._key(session_id))
            if not raw:
                return []
            return _deserialize_history(raw)
        # 内存回退
        self._mem_access[session_id] = time.monotonic()
        return self._mem.get(session_id, [])

    def _save_history(self, session_id: str, history: list) -> None:
        if self.redis is not None:
            pipe = self.redis.pipeline()
            pipe.setex(self._key(session_id), SESSION_TTL_SECONDS, _serialize_history(history))
            pipe.sadd(_SESSION_INDEX_KEY, session_id)
            pipe.expire(_SESSION_INDEX_KEY, SESSION_TTL_SECONDS)
            pipe.execute()
            return
        # 内存回退
        self._mem[session_id] = history
        self._mem_access[session_id] = time.monotonic()
        self._evict_if_needed()

    def _delete_history(self, session_id: str) -> None:
        if self.redis is not None:
            self.redis.delete(self._key(session_id))
            self.redis.srem(_SESSION_INDEX_KEY, session_id)
            self.redis.delete(self._token_key(session_id))
            return
        self._mem.pop(session_id, None)
        self._mem_access.pop(session_id, None)
        self._mem_tokens.pop(session_id, None)
        with self._mem_locks_guard:
            self._mem_locks.pop(session_id, None)

    # ---------- 内存回退专用 ----------
    def _evict_if_needed(self) -> None:
        if len(self._mem) <= MAX_SESSIONS:
            return
        # 淘汰最久未活动
        oldest = min(self._mem_access, key=self._mem_access.get)
        self._mem.pop(oldest, None)
        self._mem_access.pop(oldest, None)

    def cleanup_expired(self) -> int:
        """清理过期会话。Redis 模式下由 TTL 自动过期，本方法仅清理索引残留与内存模式。"""
        if self.redis is not None:
            # 清理索引集合中已过期的成员（成员本身可能已 TTL 失效）
            members = self.redis.smembers(_SESSION_INDEX_KEY)
            removed = 0
            for sid in members:
                if not self.redis.exists(self._key(sid)):
                    self.redis.srem(_SESSION_INDEX_KEY, sid)
                    removed += 1
            return removed
        # 内存模式
        now = time.monotonic()
        expired = [sid for sid, t in self._mem_access.items() if now - t > SESSION_TTL_SECONDS]
        for sid in expired:
            self._mem.pop(sid, None)
            self._mem_access.pop(sid, None)
            self._mem_tokens.pop(sid, None)
        if expired:
            with self._mem_locks_guard:
                for sid in expired:
                    self._mem_locks.pop(sid, None)
        return len(expired)

    @property
    def active_count(self) -> int:
        if self.redis is not None:
            return self.redis.scard(_SESSION_INDEX_KEY)
        return len(self._mem)

    def clear(self) -> None:
        """进程退出时调用（优雅关停）。"""
        if self.redis is not None:
            # 仅清本服务管理的会话键与 token，不触碰其它命名空间
            members = self.redis.smembers(_SESSION_INDEX_KEY) or []
            keys = [self._key(sid) for sid in members] + \
                   [self._token_key(sid) for sid in members] + \
                   [_SESSION_INDEX_KEY]
            if keys:
                self.redis.delete(*keys)
            return
        self._mem.clear()
        self._mem_access.clear()
        self._mem_tokens.clear()
        with self._mem_locks_guard:
            self._mem_locks.clear()


async def run_cleanup_loop(manager: SessionManager, interval: int = SESSION_CLEANUP_INTERVAL):
    """后台定期清理过期会话索引，不阻塞请求热路径"""
    while True:
        await asyncio.sleep(interval)
        try:
            removed = manager.cleanup_expired()
            if removed:
                print(f"[session] 已清理 {removed} 个过期会话索引，当前活跃: {manager.active_count}")
        except Exception as e:
            print(f"[session] 清理循环异常（忽略）: {e}")

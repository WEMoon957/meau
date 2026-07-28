"""FastAPI 服务 - 为前端提供点餐智能体 API

接口：
  POST /api/ai/chat  - 对话接口，返回AI回复
  POST /api/ai/reset - 重置对话
  GET  /api/ai/info  - 获取服务信息
  GET  /api/health   - 健康检查
"""

import asyncio
import ipaddress
import io
import logging
import os
import sys
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if sys.platform == "win32":
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

logger = logging.getLogger("api_server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# 尽力识别上游 LLM 异常以返回更精确的状态码（503）；导入失败时降级为统一 500
_LLM_UPSTREAM_EXC: tuple = ()
try:
    from openai import (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        APIStatusError,
    )
    _LLM_UPSTREAM_EXC = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        APIStatusError,
    )
except ImportError:
    pass

from agent import OrderingAgent, AgentError
from menu_data import get_all_dishes
from rate_limiter import (
    ip_chat_limiter,
    session_chat_limiter,
    CHAT_RATE_PER_IP,
    CHAT_RATE_PER_SESSION,
    CHAT_RATE_WINDOW,
)
from session_manager import (
    SessionManager,
    run_cleanup_loop,
    SESSION_TTL_SECONDS,
    MAX_SESSIONS,
)
from kb_query import preload_kb


# ======================== 配置 ========================
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "qwen-turbo")
MAX_CONCURRENT_CHATS = int(os.environ.get("MAX_CONCURRENT_CHATS", "20"))

_chat_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHATS)
_session_manager: SessionManager | None = None
_cleanup_task: asyncio.Task | None = None
_limiter_cleanup_task: asyncio.Task | None = None


def _create_agent() -> OrderingAgent:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    return OrderingAgent(api_key=api_key, model=DEFAULT_MODEL, base_url=base_url)


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager 尚未初始化")
    return _session_manager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _session_manager, _cleanup_task, _limiter_cleanup_task

    _session_manager = SessionManager(_create_agent)
    get_all_dishes()

    try:
        preload_kb()
    except Exception as e:
        print(f"⚠️ 菜品知识库预加载失败，首次知识库检索时将重试: {e}")

    try:
        from dish_rules import preload_rules
        preload_rules()
    except Exception as e:
        print(f"⚠️ 菜品规则引擎预加载失败，推荐时将重试: {e}")

    _cleanup_task = asyncio.create_task(run_cleanup_loop(_session_manager))
    _limiter_cleanup_task = asyncio.create_task(_run_limiter_cleanup())

    yield

    if _cleanup_task:
        _cleanup_task.cancel()
    if _limiter_cleanup_task:
        _limiter_cleanup_task.cancel()
    if _session_manager:
        _session_manager.clear()


async def _run_limiter_cleanup():
    """定期清理限流器过期 key，防止内存泄漏"""
    while True:
        await asyncio.sleep(600)
        ip_chat_limiter.cleanup_stale()
        session_chat_limiter.cleanup_stale()


# ======================== FastAPI 应用 ========================
app = FastAPI(title="小味点餐智能体 API", lifespan=lifespan)


def _load_cors_origins() -> list[str]:
    """从环境变量加载允许的跨域来源。

    生产环境必须显式设置 CORS_ALLOWED_ORIGINS（逗号分隔），
    例如：https://order.example.com,https://www.example.com
    未配置时默认仅放行本地开发来源，拒绝任意第三方站点调用。
    """
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "null",  # 允许 file:// 协议（桌面双击 HTML）访问
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)


# ======================== 请求/响应模型 ========================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=64)


class ChatResponse(BaseModel):
    code: int
    msg: str
    aimessage: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str = Field(default="", max_length=64)


class ResetResponse(BaseModel):
    code: int
    msg: str
    session_id: str


def _resolve_session_id(session_id: str, *, auto_create: bool = True) -> str:
    sid = session_id.strip()
    if sid:
        return sid
    if auto_create:
        return str(uuid.uuid4())
    return "default"


def _load_trusted_proxy_networks() -> list:
    """加载可信反向代理 CIDR 列表（逗号分隔），例如 127.0.0.1/32,10.0.0.0/8。

    仅当直连来源属于可信代理时，才信任其转发的 X-Forwarded-For；
    未配置时一律忽略 XFF，直接使用 TCP 对端地址（fail-safe）。
    """
    raw = os.environ.get("TRUSTED_PROXIES", "")
    networks = []
    for cidr in raw.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            logger.warning("忽略无效的 TRUSTED_PROXIES 条目: %s", cidr)
    return networks


_TRUSTED_PROXY_NETS = _load_trusted_proxy_networks()


def _is_trusted_proxy(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in _TRUSTED_PROXY_NETS)


def _client_ip(request: Request) -> str:
    """解析真实客户端 IP。

    安全策略：
    1. TCP 对端地址（request.client.host）是最可信的信号。
    2. 仅当直连来源是已配置的可信代理时，才解析 X-Forwarded-For；
       从右向左跳过可信代理跳，取第一个非可信地址作为真实客户端。
    3. 未配置可信代理时，完全忽略客户端可伪造的 XFF，使用 TCP 对端地址。
    """
    direct_ip = request.client.host if request.client else None

    if direct_ip and _is_trusted_proxy(direct_ip):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            for ip in reversed(parts):
                if not _is_trusted_proxy(ip):
                    return ip
            return parts[-1] if parts else direct_ip

    return direct_ip or "unknown"


def _check_chat_rate_limit(request: Request, session_id: str) -> None:
    """O(1) 限流检查，超限立即 429，不进入 LLM 调用"""
    ip = _client_ip(request)

    allowed, retry_after = session_chat_limiter.allow(session_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"会话请求过于频繁，请 {retry_after} 秒后重试",
            headers={"Retry-After": str(retry_after)},
        )

    allowed, retry_after = ip_chat_limiter.allow(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"IP 请求过于频繁，请 {retry_after} 秒后重试",
            headers={"Retry-After": str(retry_after)},
        )


# ======================== 接口 ========================
@app.post("/api/ai/chat", response_model=ChatResponse)
async def ai_chat(request: ChatRequest, http_request: Request):
    """对话接口，发送用户消息，返回AI回复"""
    session_id = _resolve_session_id(request.session_id)
    _check_chat_rate_limit(http_request, session_id)

    manager = get_session_manager()
    agent = manager.get_agent(session_id)

    if _chat_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="服务繁忙，请稍后重试",
            headers={"Retry-After": "5"},
        )

    try:
        async with _chat_semaphore:
            aimessage = await asyncio.to_thread(agent.chat, request.message.strip())
        manager.touch(session_id)
        return {
            "code": 200,
            "msg": "success",
            "aimessage": aimessage,
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except AgentError as e:
        logger.warning("chat 上游模型异常 session_id=%s: %s", session_id, e)
        raise HTTPException(status_code=502, detail="模型服务暂时不可用，请稍后重试")
    except Exception as e:
        logger.exception("chat 处理失败 session_id=%s", session_id)
        if _LLM_UPSTREAM_EXC and isinstance(e, _LLM_UPSTREAM_EXC):
            raise HTTPException(
                status_code=503,
                detail="模型服务繁忙，请稍后重试",
                headers={"Retry-After": "5"},
            )
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.post("/api/ai/reset", response_model=ResetResponse)
async def ai_reset(request: ResetRequest):
    """重置对话，清空历史上下文"""
    session_id = _resolve_session_id(request.session_id, auto_create=False)
    manager = get_session_manager()
    agent = manager.get_agent(session_id)
    agent.reset()
    manager.touch(session_id)
    return {"code": 200, "msg": "success", "session_id": session_id}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    manager = get_session_manager()
    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "dish_count": len(get_all_dishes()),
            "active_sessions": manager.active_count,
            "max_sessions": MAX_SESSIONS,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "max_concurrent_chats": MAX_CONCURRENT_CHATS,
            "concurrent_at_capacity": _chat_semaphore.locked(),
            "rate_limits": {
                "per_session": f"{CHAT_RATE_PER_SESSION}/{CHAT_RATE_WINDOW}s",
                "per_ip": f"{CHAT_RATE_PER_IP}/{CHAT_RATE_WINDOW}s",
            },
        },
    }


@app.get("/api/ai/info")
async def ai_info():
    """获取服务信息"""
    from tools import ALL_TOOLS
    return {
        "code": 200,
        "msg": "success",
        "data": {
            "name": "小味点餐智能体",
            "model": DEFAULT_MODEL,
            "tools": [t.name for t in ALL_TOOLS],
            "capabilities": [
                "菜品问答", "智能推荐",
                "菜品知识库查询", "搭配方案推荐", "互斥规则提示", "水果过敏原查询"
            ],
            "limits": {
                "session_ttl_seconds": SESSION_TTL_SECONDS,
                "max_sessions": MAX_SESSIONS,
                "max_concurrent_chats": MAX_CONCURRENT_CHATS,
                "chat_rate_per_session": CHAT_RATE_PER_SESSION,
                "chat_rate_per_ip": CHAT_RATE_PER_IP,
                "chat_rate_window_seconds": CHAT_RATE_WINDOW,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

"""FastAPI 服务 - 为前端提供点餐智能体 API

接口：
  POST /api/ai/session - 签发会话 token（认证入口）
  POST /api/ai/chat    - 对话接口，返回AI回复（需携带 token）
  POST /api/ai/reset   - 重置对话（需携带 token，仅可操作自己的会话）
  GET  /api/ai/info    - 获取服务信息
  GET  /api/health     - 健康检查

认证模型：
  客户端先调用 /api/ai/session 获取 {session_id, token}，
  后续请求在 Authorization: Bearer <token> 头中携带 token。
  session_id 由服务端签发并经 HMAC 签名绑定到 token，
  拒绝客户端自定义，token 持有者即 session 所有者。
"""

import asyncio
import io
import os
import sys

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

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from agent import OrderingAgent
from auth import (
    SESSION_TOKEN_TTL,
    create_session_token,
    extract_bearer_token,
    new_session_id,
    verify_session_token,
)
from menu_data import get_all_dishes
from rate_limiter import (
    ip_chat_limiter,
    session_chat_limiter,
    session_create_limiter,
    CHAT_RATE_PER_IP,
    CHAT_RATE_PER_SESSION,
    CHAT_RATE_WINDOW,
    SESSION_CREATE_PER_IP,
    SESSION_CREATE_WINDOW,
)
from session_manager import (
    SessionManager,
    run_cleanup_loop,
    SESSION_TTL_SECONDS,
    MAX_SESSIONS,
)
from tools import set_session_id, reset_cart
from vector_store import preload_vector_store


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

    if not os.environ.get("SESSION_SECRET"):
        print("⚠️ 严重：SESSION_SECRET 未配置，所有需要认证的接口（/api/ai/chat、/api/ai/reset）将拒绝请求。")
        print("⚠️ 请在 .env 中设置 SESSION_SECRET（建议使用 `openssl rand -hex 32` 生成）。")

    _session_manager = SessionManager(_create_agent)
    get_all_dishes()
    try:
        preload_vector_store()
    except Exception as e:
        print(f"⚠️ 向量库预加载失败，首次话术检索时将重试: {e}")

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
        session_create_limiter.cleanup_stale()


# ======================== FastAPI 应用 ========================
app = FastAPI(title="小味点餐智能体 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================== 请求/响应模型 ========================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    # session_id 字段已废弃：会话 ID 现由服务端签发的 token 绑定，
    # 客户端无需也无法通过请求体指定。保留字段仅为向后兼容，服务端将忽略其值。
    session_id: str = Field(default="", max_length=64, deprecated=True)


class ChatResponse(BaseModel):
    code: int
    msg: str
    aimessage: str
    session_id: str


class ResetRequest(BaseModel):
    # 同上，已废弃，由 token 绑定的 session_id 决定要重置的会话
    session_id: str = Field(default="", max_length=64, deprecated=True)


class ResetResponse(BaseModel):
    code: int
    msg: str
    session_id: str


class SessionResponse(BaseModel):
    code: int
    msg: str
    session_id: str
    token: str


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def require_session(authorization: str = Header(default="")) -> str:
    """认证依赖：校验 Authorization: Bearer <token>，返回绑定的 session_id

    - 缺失/无效/过期 token 一律返回 401
    - session_id 由服务端签发并经 HMAC 签名绑定，客户端无法伪造或自定义
    - token 持有者即 session 所有者，/api/ai/reset 等敏感操作天然受限为自有会话
    """
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="缺少认证 token，请先调用 POST /api/ai/session 获取",
            headers={"WWW-Authenticate": "Bearer"},
        )
    ok, result = verify_session_token(token)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail=result,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return result


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
@app.post("/api/ai/session", response_model=SessionResponse)
async def ai_session(http_request: Request):
    """签发新的会话 token（认证入口）

    返回 {session_id, token}。客户端需在后续 /api/ai/chat、/api/ai/reset
    请求的 Authorization: Bearer <token> 头中携带 token。
    session_id 由服务端签发并经 HMAC 签名绑定，客户端无法伪造或自定义。
    """
    ip = _client_ip(http_request)
    allowed, retry_after = session_create_limiter.allow(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"会话创建过于频繁，请 {retry_after} 秒后重试",
            headers={"Retry-After": str(retry_after)},
        )

    session_id = new_session_id()
    try:
        token = create_session_token(session_id)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"code": 200, "msg": "success", "session_id": session_id, "token": token}


@app.post("/api/ai/chat", response_model=ChatResponse)
async def ai_chat(
    request: ChatRequest,
    http_request: Request,
    session_id: str = Depends(require_session),
):
    """对话接口，发送用户消息，返回AI回复

    需携带有效 token；session_id 来自 token 绑定，请求体中的 session_id 字段被忽略。
    """
    _check_chat_rate_limit(http_request, session_id)

    set_session_id(session_id)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/reset", response_model=ResetResponse)
async def ai_reset(session_id: str = Depends(require_session)):
    """重置对话，清空历史上下文与购物车

    需携带有效 token；仅可重置 token 绑定的自有会话，无法操作他人会话。
    """
    set_session_id(session_id)
    manager = get_session_manager()
    agent = manager.get_agent(session_id)
    agent.reset()
    reset_cart(session_id)
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
            "auth": {
                "session_token_ttl_seconds": SESSION_TOKEN_TTL,
                "session_secret_configured": bool(os.environ.get("SESSION_SECRET")),
            },
            "rate_limits": {
                "per_session": f"{CHAT_RATE_PER_SESSION}/{CHAT_RATE_WINDOW}s",
                "per_ip": f"{CHAT_RATE_PER_IP}/{CHAT_RATE_WINDOW}s",
                "session_create_per_ip": f"{SESSION_CREATE_PER_IP}/{SESSION_CREATE_WINDOW}s",
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
                "菜品问答", "智能推荐", "购物车管理",
                "辅助下单", "服务员话术生成"
            ],
            "auth": {
                "scheme": "Bearer (HMAC-SHA256 signed token)",
                "session_token_ttl_seconds": SESSION_TOKEN_TTL,
                "session_issuance_endpoint": "/api/ai/session",
            },
            "limits": {
                "session_ttl_seconds": SESSION_TTL_SECONDS,
                "max_sessions": MAX_SESSIONS,
                "max_concurrent_chats": MAX_CONCURRENT_CHATS,
                "chat_rate_per_session": CHAT_RATE_PER_SESSION,
                "chat_rate_per_ip": CHAT_RATE_PER_IP,
                "chat_rate_window_seconds": CHAT_RATE_WINDOW,
                "session_create_per_ip": SESSION_CREATE_PER_IP,
                "session_create_window_seconds": SESSION_CREATE_WINDOW,
            },
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

"""FastAPI 服务 - 为前端提供点餐智能体 API

接口：
  POST /api/ai/chat  - 对话接口，返回AI回复
  POST /api/ai/reset - 重置对话
  GET  /api/ai/info  - 获取服务信息
  GET  /api/health   - 健康检查
"""

import asyncio
import io
import os
import secrets
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

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from agent import OrderingAgent
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
from tools import set_session_id, reset_cart
from vector_store import preload_vector_store


# ======================== 配置 ========================
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "qwen-turbo")
MAX_CONCURRENT_CHATS = int(os.environ.get("MAX_CONCURRENT_CHATS", "20"))

# 管理端点认证 token（管理类写操作需通过 X-Admin-Token 校验）
# 未配置时管理端点直接禁用，拒绝所有请求
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN", "")

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


class AddScriptRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    script_type: str = Field(default="custom", max_length=32)
    dish_name: str = Field(default="", max_length=100)
    scene: str = Field(default="", max_length=200)


def _require_admin(x_admin_token: str = Header(default="")) -> None:
    """管理端点认证：校验 X-Admin-Token 与配置的 ADMIN_API_TOKEN（常量时间比较）"""
    if not ADMIN_API_TOKEN:
        # 未配置管理 token 时，管理端点直接禁用
        raise HTTPException(status_code=503, detail="管理端点未启用（未配置 ADMIN_API_TOKEN）")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_API_TOKEN):
        raise HTTPException(status_code=401, detail="未授权的管理请求")


def _resolve_session_id(session_id: str, *, auto_create: bool = True) -> str:
    sid = session_id.strip()
    if sid:
        return sid
    if auto_create:
        return str(uuid.uuid4())
    return "default"


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


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
async def ai_reset(request: ResetRequest):
    """重置对话，清空历史上下文与购物车"""
    session_id = _resolve_session_id(request.session_id, auto_create=False)
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
                "菜品问答", "智能推荐", "购物车管理",
                "辅助下单", "服务员话术生成"
            ],
            "limits": {
                "session_ttl_seconds": SESSION_TTL_SECONDS,
                "max_sessions": MAX_SESSIONS,
                "max_concurrent_chats": MAX_CONCURRENT_CHATS,
                "chat_rate_per_session": CHAT_RATE_PER_SESSION,
                "chat_rate_per_ip": CHAT_RATE_PER_IP,
                "chat_rate_window_seconds": CHAT_RATE_WINDOW,
            },
        }
    }


@app.post("/api/admin/script")
async def admin_add_script(
    payload: AddScriptRequest,
    _: None = Depends(_require_admin),
):
    """管理端点：向话术向量库添加自定义话术（需 X-Admin-Token 认证）

    该写操作不暴露给匿名 Agent，仅限持有管理 token 的调用方使用，
    避免匿名用户通过提示词诱导 LLM 写入持久化投毒内容。
    """
    from vector_store import add_script

    try:
        msg = add_script(
            payload.content,
            payload.script_type,
            payload.dish_name,
            payload.scene,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"code": 200, "msg": msg}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

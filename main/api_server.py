"""FastAPI 服务 - 为前端提供点餐智能体 API

接口：
  POST /api/ai/chat  - 对话接口，返回AI回复
  POST /api/ai/reset - 重置对话
  GET  /api/ai/info  - 获取服务信息
  GET  /api/health   - 健康检查（含 DB/KB/Redis 探活）

并发模型说明（高并发改造后）：
  - 多 worker：通过 gunicorn + UvicornWorker 横向扩展（见 gunicorn_conf.py / deploy.sh）。
  - 会话/限流：外部化到 Redis，多 worker 共享，杜绝会话串号与限流被 worker 数倍绕过。
  - 共享无状态 Agent：每个 worker 持有一个 OrderingAgent，所有会话复用 graph（只读、可并发）。
  - 背压：_chat_semaphore 限制单 worker 并发 LLM 调用，超限立即 503，避免请求堆积。
  - 优雅关停：lifespan 关闭时关闭 DB 连接池与 Redis 连接，gunicorn graceful_timeout 配合 drain。
"""

import asyncio
import ipaddress
import io
import logging
import os
import sys
import uuid

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
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
import rate_limiter
from rate_limiter import (
    init_limiters,
    CHAT_RATE_PER_IP,
    CHAT_RATE_PER_SESSION,
    CHAT_RATE_WINDOW,
)
from session_manager import (
    SessionManager,
    SessionBusyError,
    run_cleanup_loop,
    SESSION_TTL_SECONDS,
    MAX_SESSIONS,
)
from kb_query import preload_kb
import db as _db


# ======================== 配置 ========================
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "qwen3.7-max")
MAX_CONCURRENT_CHATS = int(os.environ.get("MAX_CONCURRENT_CHATS", "20"))
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

_chat_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHATS)
_session_manager: SessionManager | None = None
_cleanup_task: asyncio.Task | None = None
_limiter_cleanup_task: asyncio.Task | None = None
_redis_client = None


def _create_redis_client():
    """创建 Redis 客户端。未配置 REDIS_URL 时返回 None（回退单 worker 内存模式）。"""
    if not REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        client.ping()
        logger.info("Redis 已连接: %s", REDIS_URL)
        return client
    except Exception as e:
        # Redis 是多 worker 的硬前提；连不上直接退出，避免以"伪多 worker"模式运行导致串号
        raise RuntimeError(f"REDIS_URL 已配置但连接失败，拒绝以多 worker 模式启动: {e}") from e


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
    global _session_manager, _cleanup_task, _limiter_cleanup_task, _redis_client

    # 1. Redis（多 worker 共享会话/限流的前提）
    _redis_client = _create_redis_client()

    # 2. 限流器（Redis 或内存）
    init_limiters(_redis_client)

    # 3. 共享无状态 Agent + 会话管理器
    agent = _create_agent()
    _session_manager = SessionManager(agent, redis_client=_redis_client)

    # 4. 预热数据
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

    # 5. 后台清理任务
    _cleanup_task = asyncio.create_task(run_cleanup_loop(_session_manager))
    _limiter_cleanup_task = asyncio.create_task(_run_limiter_cleanup())

    yield

    # 6. 优雅关停：取消后台任务 -> 清理会话索引 -> 关闭 DB 池/Redis
    if _cleanup_task:
        _cleanup_task.cancel()
    if _limiter_cleanup_task:
        _limiter_cleanup_task.cancel()
    if _session_manager:
        try:
            _session_manager.clear()
        except Exception as e:
            logger.warning("会话清理失败（忽略）: %s", e)
    _db.close_pool()
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception:
            pass


async def _run_limiter_cleanup():
    """定期清理限流器过期 key，防止内存泄漏（Redis 模式下为 no-op）"""
    while True:
        await asyncio.sleep(600)
        try:
            if rate_limiter.ip_chat_limiter is not None:
                rate_limiter.ip_chat_limiter.cleanup_stale()
            if rate_limiter.session_chat_limiter is not None:
                rate_limiter.session_chat_limiter.cleanup_stale()
        except Exception as e:
            print(f"[limiter] 清理异常（忽略）: {e}")


# ======================== FastAPI 应用 ========================
app = FastAPI(title="小菌点餐智能体 API", lifespan=lifespan)


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
        "http://172.16.11.82:3000",
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
    session_id: str = Field(..., min_length=1, max_length=64)
    membership_level: str = Field(default="普通会员", max_length=32)
    user_message: str = Field(..., min_length=1, max_length=2000)


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
    """O(1) 限流检查，超限立即 429，不进入 LLM 调用。

    Redis 故障时抛 503（不静默放行，防 LLM 被打爆）。
    """
    ip = _client_ip(request)

    # 通过模块属性访问：init_limiters() 在 lifespan 中重新赋值模块级实例，
    # 按名导入会绑定到旧值（None），故这里走 rate_limiter.<name> 取最新实例。
    ses_limiter = rate_limiter.session_chat_limiter
    ip_limiter = rate_limiter.ip_chat_limiter
    if ses_limiter is None or ip_limiter is None:
        raise HTTPException(
            status_code=503,
            detail="限流服务尚未就绪，请稍后重试",
            headers={"Retry-After": "5"},
        )

    try:
        allowed, retry_after = ses_limiter.allow(session_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"会话请求过于频繁，请 {retry_after} 秒后重试",
                headers={"Retry-After": str(retry_after)},
            )

        allowed, retry_after = ip_limiter.allow(ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"IP 请求过于频繁，请 {retry_after} 秒后重试",
                headers={"Retry-After": str(retry_after)},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("限流器检查异常（疑似 Redis 故障），拒绝请求以保护 LLM")
        raise HTTPException(
            status_code=503,
            detail="限流服务暂不可用，请稍后重试",
            headers={"Retry-After": "5"},
        ) from e


# ======================== 接口 ========================
@app.post("/api/ai/chat", response_model=ChatResponse)
async def ai_chat(request: ChatRequest, http_request: Request):
    """对话接口，发送用户消息，返回AI回复"""
    session_id = _resolve_session_id(request.session_id)
    _check_chat_rate_limit(http_request, session_id)

    manager = get_session_manager()

    if _chat_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="服务繁忙，请稍后重试",
            headers={"Retry-After": "5"},
        )

    try:
        async with _chat_semaphore:
            # 无状态 agent + Redis 历史：在线程池中执行（graph.invoke 为同步阻塞调用）
            aimessage = await asyncio.to_thread(
                manager.chat,
                session_id,
                request.user_message.strip(),
                request.membership_level.strip(),
            )
        # API 层再做一次兜底清理：防止 agent 内部任何出口漏网的 DSML / think 块 /
        # 控制字符跑到前端渲染成方框 □。此处若被清空仍按正常响应返回（空回复）。
        aimessage = OrderingAgent._strip_dsml(aimessage)
        return {
            "code": 200,
            "msg": "success",
            "aimessage": aimessage,
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except SessionBusyError:
        raise HTTPException(
            status_code=429,
            detail="上一条消息还在处理中，请稍后重试",
            headers={"Retry-After": "3"},
        )
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
    manager.reset(session_id)
    return {"code": 200, "msg": "success", "session_id": session_id}


@app.get("/api/health")
async def health_check():
    """健康检查（含 DB/KB/Redis 探活，供负载均衡判断）"""
    manager = get_session_manager()

    # DB 探活（池内取连接执行 SELECT 1）
    db_ok = True
    try:
        _db.test_connection()
    except Exception:
        db_ok = False

    # Redis 探活（未启用视为 ok 并标记未启用）
    redis_ok = True
    if _redis_client is not None:
        try:
            _redis_client.ping()
        except Exception:
            redis_ok = False

    # KB 探活：仅检测单例是否加载，不在健康检查里触发向量检索（避免拖慢探活）
    kb_ok = True
    try:
        from kb_query import _kb_instance  # noqa: F401
        kb_ok = _kb_instance is not None
    except Exception:
        kb_ok = False

    # 任一关键依赖不可用 -> 503，让 LB 摘流
    degraded = not (db_ok and redis_ok)
    status_code = 503 if degraded else 200

    return {
        "code": status_code,
        "msg": "ok" if not degraded else "degraded",
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
            "dependencies": {
                "db": "ok" if db_ok else "fail",
                "redis": "ok" if redis_ok else ("disabled" if _redis_client is None else "fail"),
                "kb": "ok" if kb_ok else "unloaded",
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


# ======================== 购物车代理接口（独立模块，不动现有代码） ========================
# 全来店（收钱吧）加购代理：前端调 /api/cart/add，后端完成 RSA2 签名后转发
# 详细实现见 main/cart_api.py
from cart_api import CartAddRequest as _CartAddRequest, CartAddResponse as _CartAddResponse, add_to_cart as _add_to_cart  # noqa: E402


@app.post("/api/cart/add", response_model=_CartAddResponse)
async def cart_add(request: _CartAddRequest):
    """加入购物车 - 代理调用收钱吧 openApi（RSA2 签名在后端完成）

    前端只需传简化字段，后端补全为收钱吧完整格式并签名后转发。
    私钥只在后端，前端无需持有密钥。
    """
    return await _add_to_cart(request)


# 前端静态文件
from fastapi.responses import FileResponse

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..")


@app.get("/")
@app.get("/index.html")
async def serve_index():
    """提供前端入口页面"""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

if __name__ == "__main__":
    import uvicorn
    # 本地开发：单 worker。生产部署请使用 gunicorn（见 gunicorn_conf.py / deploy.sh）
    # 监听地址与端口固定从 .env 读取，避免每次重启手敲命令行参数
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "3000")),
    )

"""收钱吧购物车 MCP 客户端 —— 通过 stdio 子进程调用桌面 cart_server.py

架构：
  api_server / cart_agent
        │  call_cart_tools(tasks)（async）/ call_cart_tools_sync(tasks)（同步）
        ▼
  本模块：一次性 stdio 会话（spawn 子进程 → MCP initialize → 顺序 call_tool → 关闭）
        ▼
  桌面 MCP Server（cart_server.py，FastMCP stdio 传输）
        ▼
  sqb_mcp.sqb_client.call_api（RSA2 签名 → /api/v1 网关，已验证全链路可用）

可用 MCP 工具（与桌面 cart_server.py 的 @mcp.tool 一一对应）：
  - get_customer_token(group_id, customer_phone, customer_name, client_type)
      登录换取 token（加购、查购物车的前置步骤）
  - add_to_cart(group_id, org_id, goods_id, sku_id, token, goods_num, ...)
      商品加入购物车
  - query_cart(group_id, org_id, token, ...)
      查询购物车明细与金额

配置（环境变量，默认指向桌面 MCP 目录）：
  SQB_MCP_CART_SERVER  cart_server.py 脚本路径
  SQB_MCP_PYTHON       启动子进程的 Python 解释器（默认自动探测：优先 PATH 中的
                       python —— 装有官方 mcp SDK；本项目 venv 的 mcp 为定制版
                       无 fastmcp，不能启动桌面脚本，仅作最后候选）

设计说明：
  - 一次性会话而非长连接：购物车是低频写操作（会话限流 10 次/分），
    每次请求 spawn 子进程（约 1s）完全可接受；避免长连接在多 worker、
    子进程意外死亡、event loop 绑定等场景下的状态管理复杂度。
  - 批量加购在一次会话内完成「token + N 件商品」，子进程只启动一次。
  - 子进程环境为 MCP SDK 白名单默认环境（不含主进程的 SQB_* 变量），
    cart_server 会读取自己目录下的 .env（内含已验证可用的 appId/私钥/网关），
    与项目 .env 中的旧配置（网关 /api/1.0，method 404）互不干扰。
  - 同步版用 asyncio.run：仅供无事件循环的线程上下文（asyncio.to_thread）
    调用，勿在 async 函数里调用同步版。
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("mcp_cart_client")

# ======================== 配置 ========================
DEFAULT_CART_SERVER = r"c:/Users/work/Desktop/mcp/cart_server.py"
CART_SERVER_SCRIPT = os.environ.get("SQB_MCP_CART_SERVER", DEFAULT_CART_SERVER)

# 解释器优先级：
#   1. 环境变量 SQB_MCP_PYTHON 显式指定
#   2. PATH 中的 python（桌面 MCP 的 mcp.json 即用 "python" 启动，为官方 mcp SDK）
#   3. 当前解释器（sys.executable）
# 候选必须能 import mcp.server.fastmcp（官方 SDK 结构）。注意：本项目 venv 内
# 的 mcp 是定制版（无 fastmcp 模块），不能用来启动桌面 cart_server.py。
_INTERPRETER_CACHE: str | None = None


def _interpreter_ok(python: str) -> bool:
    """验证解释器能否导入官方 mcp SDK 的 FastMCP（subprocess 探测，一次成本 ~1s）"""
    try:
        return subprocess.run(
            [python, "-c", "import mcp.server.fastmcp"],
            capture_output=True, timeout=30,
            cwd=os.path.dirname(os.path.abspath(CART_SERVER_SCRIPT)),
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _resolve_interpreter() -> str:
    """解析启动桌面 cart_server.py 的 Python 解释器（结果缓存）"""
    global _INTERPRETER_CACHE
    if _INTERPRETER_CACHE:
        return _INTERPRETER_CACHE

    candidates: list[str] = []
    env_py = os.environ.get("SQB_MCP_PYTHON", "").strip()
    if env_py:
        candidates.append(env_py)
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    if sys.executable not in candidates:
        candidates.append(sys.executable)

    for py in candidates:
        if _interpreter_ok(py):
            _INTERPRETER_CACHE = py
            return py

    raise McpCartClientError(
        "找不到能运行 mcp.server.fastmcp 的 Python 解释器"
        "（可用环境变量 SQB_MCP_PYTHON 指定装有官方 mcp SDK 的解释器）"
    )

# MCP 工具名（与桌面 cart_server.py 保持一致）
TOOL_GET_CUSTOMER_TOKEN = "get_customer_token"
TOOL_ADD_TO_CART = "add_to_cart"
TOOL_QUERY_CART = "query_cart"

# 子进程启动 + MCP 握手的固定开销预算（秒）
HANDSHAKE_BUDGET = 15.0
# 单次工具调用（网关往返）的预算（秒）
PER_TOOL_BUDGET = 15.0


class McpCartClientError(RuntimeError):
    """MCP 购物车客户端调用异常。"""


def _server_params() -> StdioServerParameters:
    """构造 stdio 子进程启动参数（cwd 必须是脚本目录，保证 sqb_mcp 包可导入）"""
    if not os.path.isfile(CART_SERVER_SCRIPT):
        raise McpCartClientError(
            f"cart_server.py 不存在: {CART_SERVER_SCRIPT}（可用环境变量 SQB_MCP_CART_SERVER 指定）"
        )
    return StdioServerParameters(
        command=_resolve_interpreter(),
        args=[CART_SERVER_SCRIPT],
        cwd=os.path.dirname(os.path.abspath(CART_SERVER_SCRIPT)),
    )


def _parse_tool_result(tool_name: str, result: Any) -> dict:
    """把 CallToolResult 解析为 dict。

    工具返回体是网关 JSON（如 {"code":"0000","msg":"success","data":{...}}），
    以文本形式放在 content[0].text 中；isError=True 时包装为 {"error": ...}。
    """
    if getattr(result, "isError", False):
        texts = [
            getattr(c, "text", "") for c in (result.content or [])
            if getattr(c, "type", "") == "text"
        ]
        raise McpCartClientError(f"MCP 工具 {tool_name} 返回错误: {' '.join(texts)[:500]}")
    for c in result.content or []:
        if getattr(c, "type", "") == "text":
            try:
                parsed = json.loads(c.text)
            except (json.JSONDecodeError, TypeError):
                return {"error": f"非 JSON 响应: {c.text[:500]}"}
            return parsed if isinstance(parsed, dict) else {"data": parsed}
    return {"error": f"MCP 工具 {tool_name} 未返回任何文本"}


async def call_cart_tools(
    tasks: list[tuple[str, dict]],
    *,
    per_tool_timeout: float = PER_TOOL_BUDGET,
) -> list[dict]:
    """在一次性 stdio 会话中顺序执行多个 MCP 工具调用。

    Args:
        tasks: [(tool_name, arguments), ...]，在同一子进程会话内按序执行。
            典型用法：[("get_customer_token", {...}), ("add_to_cart", {...}), ...]
        per_tool_timeout: 单个工具调用的超时（秒），含握手预算。

    Returns:
        与 tasks 等长的 dict 列表（网关 JSON 或 {"error": ...}）。

    Raises:
        McpCartClientError: 子进程启动失败 / 握手超时 / 工具执行出错。
    """
    if not tasks:
        return []

    results: list[dict] = []
    params = _server_params()

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=HANDSHAKE_BUDGET)
                for tool_name, arguments in tasks:
                    try:
                        raw = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments=arguments),
                            timeout=per_tool_timeout,
                        )
                        results.append(_parse_tool_result(tool_name, raw))
                    except asyncio.TimeoutError as exc:
                        raise McpCartClientError(
                            f"MCP 工具 {tool_name} 调用超时（>{per_tool_timeout}s）"
                        ) from exc
                    except McpCartClientError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        raise McpCartClientError(
                            f"MCP 工具 {tool_name} 调用失败: {exc}"
                        ) from exc
    except McpCartClientError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise McpCartClientError(f"MCP 会话建立失败: {exc}") from exc

    return results


def call_cart_tools_sync(
    tasks: list[tuple[str, dict]],
    *,
    per_tool_timeout: float = PER_TOOL_BUDGET,
) -> list[dict]:
    """call_cart_tools 的同步版（供 asyncio.to_thread 上下文的 CartAgent 调用）。

    仅限无线程事件循环的场景；在 async 代码中请直接 await call_cart_tools。
    """
    return asyncio.run(call_cart_tools(tasks, per_tool_timeout=per_tool_timeout))

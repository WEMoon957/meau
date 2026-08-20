"""全来店（收钱吧）购物车接口 —— 通过 MCP Server 调用（见 main/mcp_cart_client.py）

前端调本地后端 /api/cart/add、/api/cart/batch-add，后端经 MCP 客户端（stdio）
调用桌面 cart_server.py 暴露的 MCP 工具，完成「登录换 token → 加购 →（可选）查购物车」。

链路：
  前端 → api_server 路由（会话鉴权 + 限流）→ 本模块 → mcp_cart_client
       → 桌面 MCP Server（cart_server.py）→ sqb_mcp.sqb_client（RSA2 签名）
       → 收钱吧网关 /api/v1（method: applet.cart.increaseItemsShopCartApi，已验证可用）

与旧版直连网关方式的差异（旧版 /api/1.0 + headers + openapi.* method，恒 404）：
  - 网关地址   /api/v1
  - 公共参数   JSON body（非 HTTP headers）
  - method     applet.cart.increaseItemsShopCartApi / applet.customer.queryCustomerInfoApi
  - 鉴权       token（get_customer_token 换取），不再使用 openId
  - 签名/密钥  全部收在桌面 MCP Server 侧（其目录下 .env），本项目不再持有私钥

降级策略（保持不变）：
  MCP Server 不可用（脚本缺失/子进程失败/网关全失败）时自动降级为 mock
  成功响应（mocked=True），保证 推荐 → 确认 → 加购 业务链路可演示。
"""

import asyncio
import logging
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from mcp_cart_client import (
    McpCartClientError,
    TOOL_ADD_TO_CART,
    TOOL_GET_CUSTOMER_TOKEN,
    TOOL_QUERY_CART,
    call_cart_tools,
)

logger = logging.getLogger("cart_api")

# 测试联调默认集团/组织 ID（取自收钱吧文档示例，桌面 MCP 已验证可用）
DEFAULT_GROUP_ID = 101010
DEFAULT_ORG_ID = 1940340699403120719


# ======================== 前端请求/响应模型 ========================
class CartAddRequest(BaseModel):
    """前端加入购物车请求（简化版，后端补全为收钱吧完整格式）

    必填：goodsId / skuId / goodsNum / session_id / session_token
    方案 A：每个菜品单规格，skuId = goodsId = Dish.id

    身份：customer_phone / customer_name 为空时按集团匿名用户加购
          （customer_phone = groupId，收钱吧匿名规则）。

    安全：session_id + session_token 用于会话鉴权（C1 防御）。
    """
    groupId: int = Field(default=DEFAULT_GROUP_ID, description="集团 ID")
    orgId: int = Field(default=DEFAULT_ORG_ID, description="组织 ID")
    goodsId: int = Field(..., description="商品 id（全来店 goodsId）")
    skuId: int = Field(..., description="商品 SKU id（方案 A 与 goodsId 相同）")
    goodsName: str = Field(default="", max_length=100)
    goodsNum: float = Field(default=1, gt=0)
    businessType: int = Field(default=3, ge=1, le=4)  # 1自提 2外卖 3堂食 4外带
    # 会员身份（get_customer_token 换 token 用；为空走集团匿名用户）
    customer_phone: str = Field(default="", max_length=20, description="会员手机号，空则匿名")
    customer_name: str = Field(default="", max_length=50, description="会员姓名，空则匿名")
    # 会话鉴权字段（C1 防御：购物车操作必须持有有效会话凭证）
    session_id: str = Field(..., min_length=1, max_length=64, description="会话 ID")
    session_token: str = Field(..., min_length=1, max_length=128, description="会话令牌")


class CartAddResponse(BaseModel):
    code: int
    msg: str
    data: Optional[dict] = None


# ======================== 收钱吧返回解析 ========================
def _is_success(result) -> bool:
    """判断收钱吧网关返回是否成功（已验证测试网关成功码为字符串 "0000"）"""
    if not isinstance(result, dict):
        return False
    code = result.get("code")
    return code in (0, "0", 200, "200", "0000", "SUCCESS", "success")


def _extract_token(token_result: dict) -> str:
    """从 get_customer_token 的网关返回中提取 token"""
    if not isinstance(token_result, dict):
        raise McpCartClientError("get_customer_token 返回格式异常")
    if not _is_success(token_result):
        msg = token_result.get("msg") or token_result.get("error") or "登录失败"
        raise McpCartClientError(f"获取收钱吧 token 失败: {msg}")
    data = token_result.get("data") or {}
    token = data.get("token") or ""
    if not token:
        raise McpCartClientError("获取收钱吧 token 失败: 响应无 token")
    return token


def _token_args(group_id: int, customer_phone: str, customer_name: str) -> dict:
    """构造 get_customer_token 工具参数；手机号为空时按集团匿名用户规则"""
    phone = (customer_phone or "").strip() or str(group_id)
    name = (customer_name or "").strip() or "匿名用户"
    return {
        "group_id": group_id,
        "customer_phone": phone,
        "customer_name": name,
        "client_type": 5,  # 5-微信小程序
    }


# ======================== 单件加购：加入购物车 ========================
async def add_to_cart(req: CartAddRequest) -> CartAddResponse:
    """单件加入购物车：MCP 会话内完成 换 token → add_to_cart → query_cart 核对。

    MCP Server 不可用时不降级（单件接口直接报错，便于联调定位问题）。
    """
    try:
        # 第一步：换 token（单独一次 MCP 会话）
        token = _extract_token((await call_cart_tools([
            (TOOL_GET_CUSTOMER_TOKEN,
             _token_args(req.groupId, req.customer_phone, req.customer_name)),
        ]))[0])
        # 第二步：加购 + 查购物车核对（一次会话）
        results = await call_cart_tools([
            (
                TOOL_ADD_TO_CART,
                {
                    "group_id": req.groupId,
                    "org_id": req.orgId,
                    "goods_id": req.goodsId,
                    "sku_id": req.skuId,
                    "token": token,
                    "goods_num": req.goodsNum,
                    "business_type": req.businessType,
                },
            ),
            (
                TOOL_QUERY_CART,
                {
                    "group_id": req.groupId,
                    "org_id": req.orgId,
                    "token": token,
                    "business_type": req.businessType,
                },
            ),
        ])
    except McpCartClientError as e:
        logger.error("[cart] MCP 加购失败: %s", e)
        raise HTTPException(status_code=502, detail=f"收钱吧购物车服务调用失败：{e}") from e

    add_result, cart_result = results[0], results[1]
    logger.info(
        "[cart] goodsId=%s num=%s -> add code=%s msg=%s",
        req.goodsId, req.goodsNum,
        add_result.get("code"), add_result.get("msg"),
    )

    if _is_success(add_result):
        return CartAddResponse(code=200, msg="success", data={
            "add": add_result.get("data"),
            "cart": cart_result.get("data"),
        })
    return CartAddResponse(
        code=int(add_result.get("code", 500)) if str(add_result.get("code", "")).isdigit() else 500,
        msg=str(add_result.get("msg") or add_result.get("error") or "加购失败"),
        data=add_result,
    )


# ======================== 批量加购（确认下单） ========================
# 用户在对话中说"确认下单"后，前端把推荐列表里的菜品一次性加入购物车。
# 一次 MCP 会话内完成「token + N 件商品」；MCP/网关失败时降级 mock。

class BatchCartItem(BaseModel):
    """批量加购的单个菜品（前端反查 id 后组装）"""
    goodsId: int = Field(..., description="商品 id（= Dish.id）")
    skuId: int = Field(..., description="SKU id（方案 A 与 goodsId 相同）")
    goodsName: str = Field(default="", max_length=100)
    goodsNum: float = Field(default=1, gt=0)


class BatchCartAddRequest(BaseModel):
    """批量加入购物车请求

    身份：customer_phone / customer_name 为空时按集团匿名用户加购。
    """
    items: list[BatchCartItem] = Field(..., min_length=1, max_length=30)
    groupId: int = Field(default=DEFAULT_GROUP_ID)
    orgId: int = Field(default=DEFAULT_ORG_ID)
    businessType: int = Field(default=3, ge=1, le=4)  # 3 堂食
    customer_phone: str = Field(default="", max_length=20, description="会员手机号，空则匿名")
    customer_name: str = Field(default="", max_length=50, description="会员姓名，空则匿名")
    # 会话鉴权（C1 防御：批量加购同样必须持有有效会话凭证）
    session_id: str = Field(..., min_length=1, max_length=64)
    session_token: str = Field(..., min_length=1, max_length=128)


class BatchCartItemResult(BaseModel):
    goodsId: int
    goodsName: str = ""
    ok: bool
    msg: str = ""


class BatchCartAddResponse(BaseModel):
    code: int
    msg: str
    success_count: int = 0
    failed_count: int = 0
    results: list[BatchCartItemResult] = []
    mocked: bool = False
    data: Optional[dict] = None


def _build_add_task(req: BatchCartAddRequest, it: BatchCartItem, token: str) -> tuple[str, dict]:
    """构造单件商品的 add_to_cart MCP 工具任务"""
    return (
        TOOL_ADD_TO_CART,
        {
            "group_id": req.groupId,
            "org_id": req.orgId,
            "goods_id": it.goodsId,
            "sku_id": it.skuId,
            "token": token,
            "goods_num": it.goodsNum,
            "business_type": req.businessType,
        },
    )


def _mock_batch_response(req: BatchCartAddRequest, reason: str) -> BatchCartAddResponse:
    """降级 mock 成功响应（MCP/网关不可用时保证链路可演示）"""
    return BatchCartAddResponse(
        code=200,
        msg=f"success(mocked: {reason})",
        success_count=len(req.items),
        failed_count=0,
        results=[
            BatchCartItemResult(goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="mocked")
            for it in req.items
        ],
        mocked=True,
    )


async def _run_batch_via_mcp(req: BatchCartAddRequest) -> BatchCartAddResponse:
    """批量加购核心逻辑（async；sync 版通过 asyncio.run 包装复用）。

    流程：一次 MCP 会话取 token → 另一次会话批量执行 N 件加购。
    """
    # 第一步：换 token（失败直接降级 mock）
    try:
        token = _extract_token((await call_cart_tools([
            (TOOL_GET_CUSTOMER_TOKEN,
             _token_args(req.groupId, req.customer_phone, req.customer_name)),
        ]))[0])
    except McpCartClientError as e:
        logger.warning("[cart-batch] 获取 token 失败，降级 mock: %s", e)
        return _mock_batch_response(req, f"获取 token 失败: {e}")

    # 第二步：一次会话内批量加购
    tasks = [_build_add_task(req, it, token) for it in req.items]
    try:
        results = await call_cart_tools(tasks)
    except McpCartClientError as e:
        logger.warning("[cart-batch] MCP 批量加购失败，降级 mock: %s", e)
        return _mock_batch_response(req, f"MCP 调用失败: {e}")

    # 汇总
    out: list[BatchCartItemResult] = []
    success_count = 0
    for it, result in zip(req.items, results):
        if _is_success(result):
            out.append(BatchCartItemResult(
                goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="success"))
            success_count += 1
        else:
            msg = str(result.get("msg") or result.get("error") or "加购失败")
            logger.warning("[cart-batch] goodsId=%s 失败: %s", it.goodsId, msg)
            out.append(BatchCartItemResult(
                goodsId=it.goodsId, goodsName=it.goodsName, ok=False, msg=msg))

    # 全部失败 → mock 兜底（保证业务链路可走通）
    if success_count == 0:
        logger.warning("[cart-batch] 真实加购全部失败，降级为 mock 成功")
        return _mock_batch_response(req, "收钱吧网关全部失败，已降级")

    return BatchCartAddResponse(
        code=200,
        msg="success",
        success_count=success_count,
        failed_count=len(out) - success_count,
        results=out,
        mocked=False,
    )


async def batch_add_to_cart(req: BatchCartAddRequest) -> BatchCartAddResponse:
    """批量加入购物车（async 版，供 /api/cart/batch-add 路由调用）"""
    logger.info(
        "[cart-batch] items=%s group=%s org=%s phone=%s",
        len(req.items), req.groupId, req.orgId,
        req.customer_phone or "(匿名)",
    )
    return await _run_batch_via_mcp(req)


def batch_add_to_cart_sync(req: BatchCartAddRequest) -> BatchCartAddResponse:
    """批量加入购物车（同步版）。供 CartAgent 在 asyncio.to_thread 上下文调用。"""
    logger.info(
        "[cart-batch-sync] items=%s group=%s org=%s phone=%s",
        len(req.items), req.groupId, req.orgId,
        req.customer_phone or "(匿名)",
    )
    return asyncio.run(_run_batch_via_mcp(req))

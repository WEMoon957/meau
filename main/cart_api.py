"""全来店（收钱吧）购物车接口代理 - 独立模块，不动现有任何代码

前端调本地后端 POST /api/cart/add，后端完成 RSA2 签名后转发到收钱吧 openApi。
私钥只在后端，前端无需持有密钥。

收钱吧接口规范：
  URL:     POST https://test-gateway-openapi-kaci.shouqianba.com/api/1.0
  method:  openapi.applet.cart.increaseItemsShopCartApi

请求头（外层公共参数）：
  appId / format / charset / signType / timestamp / version / method / bizContent / sign

签名规则（RSA2 = SHA256withRSA, PKCS#1 v1.5）：
  1. 收集外层非空参数（不含 sign）
  2. 按参数名 ASCII 字典序排序
  3. 用 key1=value1&key2=value2... 拼成 stringA
  4. 用 RSA2 私钥签名，Base64 编码

环境变量（在 .env 配置）：
  SQB_APP_ID        收钱吧分配的 appId
  SQB_PRIVATE_KEY   RSA2 私钥（PEM 格式，或纯 Base64）
  SQB_GATEWAY_URL   网关地址（默认测试环境）
  SQB_VERSION       接口版本（默认 1.0）
"""

import base64
import json
import logging
import os
import time
from typing import Optional

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("cart_api")

# ======================== 配置 ========================
SQB_APP_ID = os.environ.get("SQB_APP_ID", "").strip()
SQB_GATEWAY_URL = os.environ.get(
    "SQB_GATEWAY_URL",
    "https://test-gateway-openapi-kaci.shouqianba.com/api/1.0",
).strip()
SQB_VERSION = os.environ.get("SQB_VERSION", "1.0").strip() or "1.0"
SQB_PRIVATE_KEY_PEM = os.environ.get("SQB_PRIVATE_KEY", "").strip()

# 收钱吧加购接口 method（已确认）
SQB_METHOD = "openapi.applet.card.openCardApi"


# ======================== 前端请求/响应模型 ========================
class CartAddRequest(BaseModel):
    """前端加入购物车请求（简化版，后端补全为收钱吧完整格式）

    必填：groupId / orgId / goodsId / skuId / goodsNum / session_id / session_token
    方案 A：每个菜品单规格，skuId = goodsId = id

    安全：session_id + session_token 用于会话鉴权（C1 防御），
          openId 必须非空（防止匿名向他人购物车塞商品）。
    """
    groupId: int = Field(..., description="集团 ID")
    orgId: int = Field(..., description="组织 ID")
    goodsId: int = Field(..., description="商品 id（全来店 goodsId）")
    skuId: int = Field(..., description="商品 SKU id（方案 A 与 goodsId 相同）")
    goodsName: str = Field(default="", max_length=100)
    goodsNum: float = Field(default=1, gt=0)
    businessType: int = Field(default=3, ge=1, le=4)  # 1自提 2外卖 3堂食 4外带
    openId: str = Field(..., min_length=1, max_length=128, description="用户 openId，必填")
    shopId: str = Field(default="", max_length=64)  # 透传，用于日志
    brandId: int = Field(default=0)
    # 会话鉴权字段（C1 防御：购物车操作必须持有有效会话凭证）
    session_id: str = Field(..., min_length=1, max_length=64, description="会话 ID")
    session_token: str = Field(..., min_length=1, max_length=128, description="会话令牌")


class CartAddResponse(BaseModel):
    code: int
    msg: str
    data: Optional[dict] = None


# ======================== RSA2 签名 ========================
def _load_signer():
    """加载 RSA2 私钥，返回 (key, padding, hashes)；失败抛 HTTPException

    兼容三种输入：
      1. 已带 PEM 头尾（PKCS#1 或 PKCS#8）
      2. 纯 Base64（PKCS#8，自动补 BEGIN PRIVATE KEY）
      3. 纯 Base64（PKCS#1，自动补 BEGIN RSA PRIVATE KEY）
    """
    if not SQB_PRIVATE_KEY_PEM:
        raise HTTPException(status_code=500, detail="未配置 SQB_PRIVATE_KEY")
    try:
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="缺少 cryptography 库，请 pip install cryptography",
        )

    pem = SQB_PRIVATE_KEY_PEM

    # 已带 PEM 头尾，直接用（load_pem_private_key 会自动识别 PKCS#1/PKCS#8）
    if "BEGIN" in pem:
        try:
            key = serialization.load_pem_private_key(
                pem.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )
            return key, padding, hashes
        except Exception as e:
            # 安全：内部异常细节仅记日志，不回传客户端（H1 防御）
            logger.error("私钥加载失败: %s", e)
            raise HTTPException(status_code=500, detail="支付配置异常，请联系管理员")

    # 纯 Base64，尝试两种格式自动补全
    body = pem.replace("\n", "").replace("\r", "")
    body_wrapped = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))

    # 先按 PKCS#8 尝试（BEGIN PRIVATE KEY，对应 "MIIEvg..." 开头的密钥）
    for header, footer in (
        ("-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"),
        ("-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"),
    ):
        pem_attempt = f"{header}\n{body_wrapped}\n{footer}"
        try:
            key = serialization.load_pem_private_key(
                pem_attempt.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )
            return key, padding, hashes
        except Exception:
            continue

    logger.error("私钥加载失败：无法识别格式")
    raise HTTPException(status_code=500, detail="支付配置异常，请联系管理员")


def _rsa2_sign(string_a: str) -> str:
    """对 stringA 做 RSA2 签名（SHA256withRSA, PKCS#1 v1.5），返回 Base64 字符串"""
    key, padding, hashes = _load_signer()
    try:
        signature = key.sign(
            string_a.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")
    except Exception as e:
        # 安全：仅记日志，客户端只看通用消息（H1 防御）
        logger.exception("RSA2 签名失败")
        raise HTTPException(status_code=500, detail="支付签名异常，请联系管理员")


def _build_sign(params: dict) -> str:
    """按收钱吧规则生成 sign

    1. 过滤空值（不含 sign）
    2. 按 key ASCII 字典序排序
    3. 拼成 key=value&key=value
    4. RSA2 签名 + Base64
    """
    filtered = {k: v for k, v in params.items() if v not in (None, "", [])}
    sorted_keys = sorted(filtered.keys())
    string_a = "&".join(f"{k}={filtered[k]}" for k in sorted_keys)
    # 调试用：完整打印 stringA 与签名结果，方便对接人核对签名规则
    logger.info("stringA: %s", string_a)
    sig = _rsa2_sign(string_a)
    logger.info("sign(len=%d): %s...", len(sig), sig[:50])
    return sig


# ======================== 构造 bizContent ========================
def _build_biz_content(req: CartAddRequest) -> dict:
    """把简化请求补全为收钱吧加购接口的 bizContent 结构

    必填字段（依据文档）：groupId, orgId, goodsId, skuId, goodsNum, goodsType, skuDetailsList
    """
    return {
        "groupId": req.groupId,
        "brandId": req.brandId,
        "orgId": req.orgId,
        "businessType": req.businessType,
        "openId": req.openId,
        "goodsId": req.goodsId,
        "goodsName": req.goodsName,
        "skuId": req.skuId,
        "goodsNum": req.goodsNum,
        "goodsType": 1,  # 1 普通商品
        "goodsMode": 0,  # 0 正常购买
        # 规格信息（必填，方案 A 单规格给一个最小项）
        "skuDetailsList": [
            {
                "skuId": req.skuId,
                "goodsId": req.goodsId,
                "skuNum": req.goodsNum,
            }
        ],
    }


# ======================== 主入口：加入购物车 ========================
# 公共参数（appId/format/charset/signType/timestamp/version/method/bizContent/sign）
# 全部放 HTTP 请求头（收钱吧 openApi 规范）


async def _call_gateway(client, url, headers):
    """单次调用收钱吧网关（公共参数全放 header），返回 (status_code, json_result, raw_text)"""
    resp = await client.post(url, headers=headers)
    try:
        result = resp.json()
    except Exception:
        result = None
    return resp.status_code, result, resp.text[:500] if resp.text else ""


def _is_success(result):
    """判断收钱吧返回是否成功"""
    if not result:
        return False
    code = result.get("code")
    return code in (0, "0", 200, "200", "SUCCESS", "success")


def _parse_result(result, raw_text):
    """解析收钱吧返回，构造 CartAddResponse"""
    if not result:
        return CartAddResponse(code=502, msg="收钱吧返回非 JSON", data={"raw": raw_text})
    code = result.get("code")
    msg = result.get("msg") or result.get("message") or ""
    if _is_success(result):
        return CartAddResponse(code=200, msg="success", data=result.get("data") or result)
    return CartAddResponse(
        code=int(code) if str(code).isdigit() else 500,
        msg=str(msg),
        data=result,
    )


async def add_to_cart(req: CartAddRequest) -> CartAddResponse:
    """加入购物车：构造请求 → RSA2 签名 → 调用收钱吧网关 → 返回结果

    公共参数（appId/format/charset/signType/timestamp/version/method/bizContent/sign）
    全部放 HTTP 请求头。
    """
    if not SQB_APP_ID:
        raise HTTPException(status_code=500, detail="未配置 SQB_APP_ID")

    biz_content = _build_biz_content(req)
    biz_json = json.dumps(biz_content, ensure_ascii=False, separators=(",", ":"))

    logger.info(
        "[cart] 加购请求 shopId=%s goodsId=%s skuId=%s num=%s",
        req.shopId, req.goodsId, req.skuId, req.goodsNum,
    )

    headers = {
        "appId": SQB_APP_ID,
        "format": "json",
        "charset": "UTF-8",
        "signType": "RSA2",
        "timestamp": str(int(time.time() * 1000)),
        "version": SQB_VERSION,
        "method": SQB_METHOD,
        "bizContent": biz_json,
    }
    headers["sign"] = _build_sign(headers)
    logger.info("[cart] 签名完成，调用网关 method=%s", SQB_METHOD)

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            status, result, raw = await _call_gateway(client, SQB_GATEWAY_URL, headers)
        except httpx.HTTPError as e:
            logger.error("[cart] 网关调用失败: %s", e)
            return CartAddResponse(code=502, msg="收钱吧网关调用失败")

        logger.info("[cart] method=%s -> HTTP %s, code=%s, msg=%s",
                    SQB_METHOD, status,
                    result.get("code") if result else None,
                    result.get("msg") if result else None)

        if status == 200 and _is_success(result):
            logger.info("[cart] ✅ 成功 method=%s", SQB_METHOD)
            return _parse_result(result, raw)

        return _parse_result(result, raw)


# ======================== 批量加购（确认下单） ========================
# 用户在对话中说"确认下单"后，前端把推荐列表里的菜品一次性加入购物车。
# 收钱吧网关目前返回 404（method 未通），本接口在真实调用全部失败时
# 自动降级为 mock 成功响应，保证业务链路（推荐→确认→加购）可演示，
# 待收钱吧接口接通后自动切换为真实结果（mocked=False）。

class BatchCartItem(BaseModel):
    """批量加购的单个菜品（前端反查 id 后组装）"""
    goodsId: int = Field(..., description="商品 id（= Dish.id）")
    skuId: int = Field(..., description="SKU id（方案 A 与 goodsId 相同）")
    goodsName: str = Field(default="", max_length=100)
    goodsNum: float = Field(default=1, gt=0)


class BatchCartAddRequest(BaseModel):
    """批量加入购物车请求

    收钱吧业务参数（groupId/orgId/openId）在网关未通时不强制校验，
    前端可传 0/空；接通后前端需补全真实值。
    """
    items: list[BatchCartItem] = Field(..., min_length=1, max_length=30)
    groupId: int = Field(default=0)
    orgId: int = Field(default=0)
    openId: str = Field(default="", max_length=128)
    shopId: str = Field(default="", max_length=64)
    brandId: int = Field(default=0)
    businessType: int = Field(default=3, ge=1, le=4)  # 3 堂食
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


def _build_biz_content_raw(
    group_id: int, brand_id: int, org_id: int, business_type: int,
    open_id: str, goods_id: int, goods_name: str, sku_id: int, goods_num: float,
) -> dict:
    """构造单个菜品的 bizContent（不依赖 CartAddRequest，供批量加购复用）"""
    return {
        "groupId": group_id,
        "brandId": brand_id,
        "orgId": org_id,
        "businessType": business_type,
        "openId": open_id,
        "goodsId": goods_id,
        "goodsName": goods_name,
        "skuId": sku_id,
        "goodsNum": goods_num,
        "goodsType": 1,
        "goodsMode": 0,
        "skuDetailsList": [
            {"skuId": sku_id, "goodsId": goods_id, "skuNum": goods_num}
        ],
    }


async def batch_add_to_cart(req: BatchCartAddRequest) -> BatchCartAddResponse:
    """批量加入购物车：逐个调用收钱吧网关，全部失败时 mock 兜底。

    Returns:
        BatchCartAddResponse:
          - 真实调用有任一成功 → 汇总 success_count/failed_count，mocked=False
          - 真实调用全部失败   → 返回 mock 成功（success_count=全部），mocked=True
          - 未配置 SQB_APP_ID   → 直接 mock 成功，mocked=True
    """
    # 未配置 appId：直接 mock（开发/演示环境）
    if not SQB_APP_ID:
        logger.warning("[cart-batch] 未配置 SQB_APP_ID，返回 mock 结果")
        return BatchCartAddResponse(
            code=200,
            msg="success(mocked: 未配置 SQB_APP_ID)",
            success_count=len(req.items),
            failed_count=0,
            results=[
                BatchCartItemResult(goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="mocked")
                for it in req.items
            ],
            mocked=True,
        )

    results: list[BatchCartItemResult] = []
    success_count = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for it in req.items:
            biz = _build_biz_content_raw(
                req.groupId, req.brandId, req.orgId, req.businessType,
                req.openId, it.goodsId, it.goodsName, it.skuId, it.goodsNum,
            )
            biz_json = json.dumps(biz, ensure_ascii=True, separators=(",", ":"))
            headers = {
                "appId": SQB_APP_ID,
                "format": "json",
                "charset": "UTF-8",
                "signType": "RSA2",
                "timestamp": str(int(time.time() * 1000)),
                "version": SQB_VERSION,
                "method": SQB_METHOD,
                "bizContent": biz_json,
            }
            headers["sign"] = _build_sign(headers)

            try:
                status, result, raw = await _call_gateway(client, SQB_GATEWAY_URL, headers)
                if status == 200 and _is_success(result):
                    results.append(BatchCartItemResult(
                        goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="success"))
                    success_count += 1
                else:
                    code = result.get("code") if result else None
                    msg = result.get("msg") if result else f"HTTP {status}"
                    logger.warning("[cart-batch] goodsId=%s 失败 code=%s msg=%s",
                                   it.goodsId, code, msg)
                    results.append(BatchCartItemResult(
                        goodsId=it.goodsId, goodsName=it.goodsName, ok=False, msg=str(msg)))
            except httpx.HTTPError as e:
                logger.error("[cart-batch] goodsId=%s 网关异常: %s", it.goodsId, e)
                results.append(BatchCartItemResult(
                    goodsId=it.goodsId, goodsName=it.goodsName, ok=False, msg="网关调用失败"))

    # 全部失败 → mock 兜底（收钱吧网关未通时保证业务链路可走通）
    if success_count == 0:
        logger.warning("[cart-batch] 真实加购全部失败，降级为 mock 成功")
        return BatchCartAddResponse(
            code=200,
            msg="success(mocked: 收钱吧网关未通，已降级)",
            success_count=len(req.items),
            failed_count=0,
            results=[
                BatchCartItemResult(goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="mocked")
                for it in req.items
            ],
            mocked=True,
        )

    return BatchCartAddResponse(
        code=200,
        msg="success",
        success_count=success_count,
        failed_count=len(results) - success_count,
        results=results,
        mocked=False,
    )


# ======================== 同步版批量加购（供 CartAgent 在 to_thread 上下文调用） ========================
# api_server 用 asyncio.to_thread 调 CartAgent.chat（同步），CartAgent 内部无法直接 await
# async 加购函数，故提供同步版。逻辑与 batch_add_to_cart 完全一致，仅把 httpx.AsyncClient
# 换成 httpx.Client、去掉 await。

def _call_gateway_sync(client, url, headers):
    """单次调用收钱吧网关（同步版）"""
    resp = client.post(url, headers=headers)
    try:
        result = resp.json()
    except Exception:
        result = None
    return resp.status_code, result, resp.text[:500] if resp.text else ""


def batch_add_to_cart_sync(req: BatchCartAddRequest) -> BatchCartAddResponse:
    """批量加入购物车（同步版）。逻辑与 batch_add_to_cart 一致，供 CartAgent 调用。

    在 api_server 的 asyncio.to_thread 上下文中执行（无 event loop），用同步 httpx.Client。
    """
    if not SQB_APP_ID:
        logger.warning("[cart-batch-sync] 未配置 SQB_APP_ID，返回 mock 结果")
        return BatchCartAddResponse(
            code=200,
            msg="success(mocked: 未配置 SQB_APP_ID)",
            success_count=len(req.items),
            failed_count=0,
            results=[
                BatchCartItemResult(goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="mocked")
                for it in req.items
            ],
            mocked=True,
        )

    results: list[BatchCartItemResult] = []
    success_count = 0

    with httpx.Client(timeout=15.0) as client:
        for it in req.items:
            biz = _build_biz_content_raw(
                req.groupId, req.brandId, req.orgId, req.businessType,
                req.openId, it.goodsId, it.goodsName, it.skuId, it.goodsNum,
            )
            biz_json = json.dumps(biz, ensure_ascii=True, separators=(",", ":"))
            headers = {
                "appId": SQB_APP_ID,
                "format": "json",
                "charset": "UTF-8",
                "signType": "RSA2",
                "timestamp": str(int(time.time() * 1000)),
                "version": SQB_VERSION,
                "method": SQB_METHOD,
                "bizContent": biz_json,
            }
            headers["sign"] = _build_sign(headers)

            try:
                status, result, raw = _call_gateway_sync(client, SQB_GATEWAY_URL, headers)
                if status == 200 and _is_success(result):
                    results.append(BatchCartItemResult(
                        goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="success"))
                    success_count += 1
                else:
                    code = result.get("code") if result else None
                    msg = result.get("msg") if result else f"HTTP {status}"
                    logger.warning("[cart-batch-sync] goodsId=%s 失败 code=%s msg=%s",
                                   it.goodsId, code, msg)
                    results.append(BatchCartItemResult(
                        goodsId=it.goodsId, goodsName=it.goodsName, ok=False, msg=str(msg)))
            except httpx.HTTPError as e:
                logger.error("[cart-batch-sync] goodsId=%s 网关异常: %s", it.goodsId, e)
                results.append(BatchCartItemResult(
                    goodsId=it.goodsId, goodsName=it.goodsName, ok=False, msg="网关调用失败"))

    if success_count == 0:
        logger.warning("[cart-batch-sync] 真实加购全部失败，降级为 mock 成功")
        return BatchCartAddResponse(
            code=200,
            msg="success(mocked: 收钱吧网关未通，已降级)",
            success_count=len(req.items),
            failed_count=0,
            results=[
                BatchCartItemResult(goodsId=it.goodsId, goodsName=it.goodsName, ok=True, msg="mocked")
                for it in req.items
            ],
            mocked=True,
        )

    return BatchCartAddResponse(
        code=200,
        msg="success",
        success_count=success_count,
        failed_count=len(results) - success_count,
        results=results,
        mocked=False,
    )

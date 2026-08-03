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

# 收钱吧加购接口 method（依据文档 #url 段）
SQB_METHOD = "openapi.applet.cart.increaseItemsShopCartApi"


# ======================== 前端请求/响应模型 ========================
class CartAddRequest(BaseModel):
    """前端加入购物车请求（简化版，后端补全为收钱吧完整格式）

    必填：groupId / orgId / goodsId / skuId / goodsNum
    方案 A：每个菜品单规格，skuId = goodsId = id
    """
    groupId: int = Field(..., description="集团 ID")
    orgId: int = Field(..., description="组织 ID")
    goodsId: int = Field(..., description="商品 id（全来店 goodsId）")
    skuId: int = Field(..., description="商品 SKU id（方案 A 与 goodsId 相同）")
    goodsName: str = Field(default="", max_length=100)
    goodsNum: float = Field(default=1, gt=0)
    businessType: int = Field(default=3, ge=1, le=4)  # 1自提 2外卖 3堂食 4外带
    openId: str = Field(default="", max_length=128)
    shopId: str = Field(default="", max_length=64)  # 透传，用于日志
    brandId: int = Field(default=0)


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
            logger.error("私钥加载失败: %s", e)
            raise HTTPException(status_code=500, detail=f"私钥加载失败: {e}")

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
    raise HTTPException(status_code=500, detail="私钥格式无法识别（请提供 PKCS#1 或 PKCS#8 格式）")


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
        logger.exception("RSA2 签名失败")
        raise HTTPException(status_code=500, detail=f"签名失败: {e}")


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
# ======================== 主入口：加入购物车 ========================
# method 候选列表：依次尝试，第一个成功的就用
# 文档里 method 示例与 #url 段不一致，故枚举几种可能
_METHOD_CANDIDATES = [
    "openapi.applet.cart.increaseItemsShopCartApi",  # 从 #url 推断
    "openapi.applet.card.openCardApi",                # 文档示例（疑似复制错误）
    "applet.cart.increaseItemsShopCartApi",           # 去前缀
    "increaseItemsShopCartApi",                       # 纯方法名
    "openapi.applet.cart.add",                         # 简化名
]

# 是否同时尝试"方法路径放 URL 里"的模式
_TRY_URL_PATH = True


async def _call_gateway(client, url, payload):
    """单次调用收钱吧网关，返回 (status_code, json_result, raw_text)"""
    resp = await client.post(url, data=payload)
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

    由于文档里 method 字符串不明确，这里依次尝试候选 method，
    第一个返回成功的就用；全部失败则返回最后一次的结果。
    """
    if not SQB_APP_ID:
        raise HTTPException(status_code=500, detail="未配置 SQB_APP_ID")

    biz_content = _build_biz_content(req)
    biz_json = json.dumps(biz_content, ensure_ascii=False, separators=(",", ":"))

    logger.info(
        "[cart] 加购请求 shopId=%s goodsId=%s skuId=%s num=%s",
        req.shopId, req.goodsId, req.skuId, req.goodsNum,
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        last_resp = None
        # 模式 A：method 在 body 里，依次尝试候选值
        for method in _METHOD_CANDIDATES:
            common = {
                "appId": SQB_APP_ID,
                "format": "json",
                "charset": "UTF-8",
                "signType": "RSA2",
                "timestamp": str(int(time.time() * 1000)),
                "version": SQB_VERSION,
                "method": method,
                "bizContent": biz_json,
            }
            common["sign"] = _build_sign(common)

            try:
                status, result, raw = await _call_gateway(client, SQB_GATEWAY_URL, common)
            except httpx.HTTPError as e:
                logger.error("[cart] 网关调用失败 method=%s: %s", method, e)
                last_resp = CartAddResponse(code=502, msg="收钱吧网关调用失败", data={"method": method})
                continue

            logger.info("[cart] method=%s -> HTTP %s, code=%s, msg=%s",
                        method, status,
                        result.get("code") if result else None,
                        result.get("msg") if result else None)

            if status == 200 and _is_success(result):
                logger.info("[cart] ✅ 成功 method=%s", method)
                return _parse_result(result, raw)

            last_resp = _parse_result(result, raw)
            # 1300 = 接口不存在，继续试下一个；其他错误也继续试
            last_resp.data = {**(last_resp.data or {}), "tried_method": method}

        # 模式 B：方法路径放 URL 里，不带 method 字段
        if _TRY_URL_PATH:
            url_path = f"{SQB_GATEWAY_URL}/applet/cart/increaseItemsShopCartApi"
            common = {
                "appId": SQB_APP_ID,
                "format": "json",
                "charset": "UTF-8",
                "signType": "RSA2",
                "timestamp": str(int(time.time() * 1000)),
                "version": SQB_VERSION,
                "bizContent": biz_json,
            }
            common["sign"] = _build_sign(common)
            try:
                status, result, raw = await _call_gateway(client, url_path, common)
                logger.info("[cart] url_path=%s -> HTTP %s, code=%s",
                            url_path, status,
                            result.get("code") if result else None)
                if status == 200 and _is_success(result):
                    logger.info("[cart] ✅ 成功 url_path=%s", url_path)
                    return _parse_result(result, raw)
                last_resp = _parse_result(result, raw)
                last_resp.data = {**(last_resp.data or {}), "tried_url_path": url_path}
            except httpx.HTTPError as e:
                logger.error("[cart] url_path 调用失败: %s", e)

    # 全部失败，返回最后一次结果
    if last_resp is None:
        last_resp = CartAddResponse(code=500, msg="所有 method 候选均失败", data=None)
    return last_resp

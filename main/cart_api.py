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
    logger.debug("stringA: %s", string_a)
    return _rsa2_sign(string_a)


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
async def add_to_cart(req: CartAddRequest) -> CartAddResponse:
    """加入购物车：构造请求 → RSA2 签名 → 调用收钱吧网关 → 返回结果"""
    if not SQB_APP_ID:
        raise HTTPException(status_code=500, detail="未配置 SQB_APP_ID")

    biz_content = _build_biz_content(req)
    biz_json = json.dumps(biz_content, ensure_ascii=False, separators=(",", ":"))

    # 外层公共参数（不含 sign）
    common = {
        "appId": SQB_APP_ID,
        "format": "json",
        "charset": "UTF-8",
        "signType": "RSA2",
        "timestamp": str(int(time.time() * 1000)),
        "version": SQB_VERSION,
        "method": SQB_METHOD,
        "bizContent": biz_json,
    }

    # 生成签名
    sign = _build_sign(common)
    common["sign"] = sign

    logger.info(
        "[cart] 加购请求 shopId=%s goodsId=%s skuId=%s num=%s",
        req.shopId, req.goodsId, req.skuId, req.goodsNum,
    )

    # 调用收钱吧网关
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(SQB_GATEWAY_URL, data=common)
    except httpx.HTTPError as e:
        logger.error("[cart] 调用收钱吧网关失败: %s", e)
        raise HTTPException(status_code=502, detail="收钱吧网关调用失败")

    if resp.status_code != 200:
        logger.error("[cart] 网关 HTTP %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(status_code=502, detail=f"收钱吧网关返回 HTTP {resp.status_code}")

    try:
        result = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="收钱吧返回非 JSON")

    # 收钱吧返回结构通常含 code/msg/data
    code = result.get("code")
    msg = result.get("msg") or result.get("message") or ""
    if code in (0, "0", 200, "200", "SUCCESS", "success"):
        return CartAddResponse(code=200, msg="success", data=result.get("data") or result)
    return CartAddResponse(
        code=int(code) if str(code).isdigit() else 500,
        msg=str(msg),
        data=result,
    )

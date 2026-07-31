"""会话认证模块 - 服务端签发签名 token，绑定 session_id

通过 HMAC-SHA256 签名的 token 实现：
1. 服务端签发 session_id，拒绝客户端自定义
2. token 持有者即 session 所有者，敏感操作（如 reset）天然鉴权
3. token 带 iat 时间戳，支持过期失效

token 格式: <base64url(payload)>.<hex_hmac>
payload: {"sid":"<uuid>","iat":<unix_ts>}

配置：
- SESSION_SECRET：HMAC 密钥，必须通过环境变量设置（建议 openssl rand -hex 32 生成）
- SESSION_TOKEN_TTL：token 有效期（秒），默认 86400（24 小时）
"""

import os
import time
import json
import base64
import hmac
import hashlib
import uuid
from typing import Optional


SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
SESSION_TOKEN_TTL = int(os.environ.get("SESSION_TOKEN_TTL", str(86400)))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(s: str) -> bytes:
    padded = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(padded.encode())


def new_session_id() -> str:
    """生成新的会话 ID（服务端签发，不接受客户端自定义）"""
    return str(uuid.uuid4())


def create_session_token(session_id: str) -> str:
    """为指定 session_id 签发签名 token

    Args:
        session_id: 服务端生成的会话 ID

    Returns:
        签名后的 token 字符串

    Raises:
        RuntimeError: SESSION_SECRET 未配置时抛出
    """
    if not SESSION_SECRET:
        raise RuntimeError("SESSION_SECRET 未配置，无法签发 token")
    payload = {"sid": session_id, "iat": int(time.time())}
    payload_b = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(SESSION_SECRET.encode(), payload_b.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b}.{sig}"


def verify_session_token(token: str) -> tuple[bool, str]:
    """校验 token

    Args:
        token: 待校验的 token 字符串

    Returns:
        (is_valid, session_id_or_error)
        - 有效时第二个返回值为 session_id
        - 无效时第二个返回值为错误描述
    """
    if not SESSION_SECRET:
        return False, "服务端未配置 SESSION_SECRET，认证不可用"
    if not token or not isinstance(token, str):
        return False, "token 为空"

    parts = token.split(".")
    if len(parts) != 2:
        return False, "token 格式错误"
    payload_b, sig = parts

    # 常量时间比较，防止计时攻击
    expected_sig = hmac.new(
        SESSION_SECRET.encode(), payload_b.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return False, "token 签名无效"

    try:
        payload = json.loads(_b64url_decode(payload_b))
    except Exception:
        return False, "token 载荷无法解析"

    sid = payload.get("sid")
    iat = payload.get("iat")
    if not isinstance(sid, str) or not sid:
        return False, "token 载荷缺少 sid"
    if not isinstance(iat, int):
        return False, "token 载荷缺少 iat"

    now = int(time.time())
    if now - iat > SESSION_TOKEN_TTL:
        return False, "token 已过期"
    # 容忍 60 秒未来时间漂移，超过则拒绝
    if iat - now > 60:
        return False, "token 时间异常"

    return True, sid


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """从 Authorization: Bearer <token> 头中提取 token

    Args:
        authorization_header: Authorization 头的原始值

    Returns:
        token 字符串；无法识别时返回 None
    """
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return None

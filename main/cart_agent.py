"""加购子 Agent - 从自然语言提取菜名并加入购物车

与 OrderingAgent（推荐/对话）平行，专门处理「加购」意图。
用户输入如"来份菌汤生态鸡子母锅和两份单点绣球菌加购"，
CartAgent 用 LLM 提取菜名+数量 → 反查 id → 同步批量加购。

设计要点：
  - 无状态：session_id/session_token 由调用方传入，不在实例中持有，
    多 worker 共享一个 CartAgent 实例，线程安全。
  - 单次 LLM 调用：仅做菜名提取（输入短、输出 JSON，通常 2-4s），
    不走多轮工具循环，比 OrderingAgent 轻。
  - 复用 OrderingAgent 的 LLM 配置（同 api_key/model/base_url）。
  - 加购走 batch_add_to_cart_sync（同步版），在 api_server 的
    asyncio.to_thread 上下文中执行。
  - 收钱吧网关未通时自动 mock 成功（mocked=True），保证链路可演示。
"""

import json
import logging
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from cart_api import (
    BatchCartAddRequest,
    BatchCartItem,
    batch_add_to_cart_sync,
)
from menu_data import find_dish_by_name

logger = logging.getLogger("cart_agent")


# 模块级单例：由 api_server.lifespan 初始化，add_to_cart 工具通过 get_cart_agent() 访问。
# 避免循环依赖：tools.py 不能从 api_server 导入，故单例放在 cart_agent 模块。
_cart_agent_instance: "CartAgent | None" = None


def init_cart_agent(agent: "CartAgent") -> None:
    """初始化模块级 CartAgent 单例（api_server.lifespan 调用）。"""
    global _cart_agent_instance
    _cart_agent_instance = agent


def get_cart_agent() -> "CartAgent":
    """获取 CartAgent 单例。未初始化时抛 RuntimeError。"""
    if _cart_agent_instance is None:
        raise RuntimeError("CartAgent 尚未初始化，请先调用 init_cart_agent")
    return _cart_agent_instance


# LLM 单次请求超时（与 OrderingAgent 一致）
CART_LLM_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "30"))


CART_SYSTEM_PROMPT = """你是加购助手，负责从顾客的自然语言中识别要加入购物车的菜品。

## 你的职责
从顾客输入中提取要加购的菜品名称和数量，返回 JSON。

## 规则（必须严格遵守）
1. **只提取顾客明确要加购的菜品**，不要编造、不要推荐、不要把"菌子""牛肉"等宽泛词当作菜名。
2. **菜名要完整**：尽量与菜单原文一致（如"菌汤生态鸡子母锅"而非"菌汤锅"）。如果顾客说的不完整，按最接近的完整菜名输出。
3. **数量识别**：顾客说"两份""3个"等数量时，填入 num；没说数量默认 1。
4. **未识别处理**：如果顾客没说要具体菜（如"把好吃的都加了""随便加"），返回空列表并给 reason。
5. **只输出 JSON**，不要任何解释、标点、markdown 修饰。

## 输出格式
{"dishes": [{"name": "完整菜名", "num": 1}, ...]}

未识别时：
{"dishes": [], "reason": "未识别到具体菜品"}

## 示例
输入："来份菌汤生态鸡子母锅和两份单点绣球菌加购"
输出：{"dishes": [{"name": "菌汤生态鸡子母锅", "num": 1}, {"name": "单点绣球菌", "num": 2}]}

输入："加一份小炒黄牛肉到购物车"
输出：{"dishes": [{"name": "小炒黄牛肉", "num": 1}]}

输入："帮我把菜单里好吃的都加了"
输出：{"dishes": [], "reason": "未识别到具体菜品"}

输入："再来一份刚才那个菌汤生态鸡子母锅"
输出：{"dishes": [{"name": "菌汤生态鸡子母锅", "num": 1}]}
"""


class CartAgentError(Exception):
    """CartAgent 处理过程中的可预期错误（如模型返回空、JSON 解析失败）。"""


class CartAgent:
    """加购子 Agent（无状态）。

    用 LLM 从自然语言提取菜名，反查菜品 id 后同步批量加购。
    单个实例可在多线程中被并发调用（无共享可变状态）。
    """

    def __init__(self, api_key: str, model: str = "qwen-max", base_url: str = ""):
        llm_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": 0,
            "timeout": CART_LLM_TIMEOUT,
            "max_retries": 1,
            "max_tokens": 1024,
        }
        # DeepSeek 模型禁用 thinking 模式（与 OrderingAgent 一致）
        if "deepseek" in model.lower():
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if base_url:
            llm_kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**llm_kwargs)

    def chat(self, user_input: str, session_id: str, session_token: str) -> tuple[str, dict]:
        """处理加购请求：LLM 提取菜名 → 反查 id → 同步加购 → 返回 (回复文本, 加购结果)。

        Args:
            user_input: 用户自然语言（如"来份菌汤生态鸡子母锅加购"）
            session_id: 会话 ID（用于加购鉴权）
            session_token: 会话令牌

        Returns:
            (reply_text, cart_result_dict)
            - reply_text: 给用户的自然语言回复
            - cart_result_dict: 加购结果（含 success_count/mocked/results），无加购时为空 dict
        """
        # 1. LLM 提取菜名
        logger.debug("cart_agent chat: input=%s", user_input[:80])
        messages = [
            SystemMessage(content=CART_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        try:
            resp = self.llm.invoke(messages)
        except Exception as e:
            logger.exception("cart_agent LLM 调用失败")
            raise CartAgentError(f"加购助手暂时不可用：{e}") from e

        content = (resp.content or "").strip()
        if not content:
            raise CartAgentError("加购助手未返回任何内容")

        # 2. 解析 JSON（容错：LLM 偶尔会包裹 markdown ```json）
        dishes = self._parse_dishes(content)
        if not dishes:
            reason = self._extract_reason(content)
            return reason or "未识别到要加购的菜品，请说明具体菜名（如：菌汤生态鸡子母锅加购）", {}

        # 3. 反查菜品 id（复用 menu_data 内存索引，精确+模糊匹配）
        resolved = []  # [{"dish": Dish, "num": int}]
        missing = []   # 未在菜单找到的菜名
        for item in dishes:
            name = item.get("name", "").strip()
            num = item.get("num", 1)
            try:
                num = max(1, int(num))
            except (TypeError, ValueError):
                num = 1
            if not name:
                continue
            d = find_dish_by_name(name)
            if d is not None:
                resolved.append({"dish": d, "num": num})
            else:
                missing.append(name)

        if not resolved:
            hint = f"未在菜单中找到：{'、'.join(missing)}" if missing else "未识别到要加购的菜品"
            return hint, {"missing": missing}

        # 4. 同步批量加购
        items = [
            BatchCartItem(
                goodsId=r["dish"].id,
                skuId=r["dish"].id,  # 方案 A：skuId = goodsId = Dish.id
                goodsName=r["dish"].name,
                goodsNum=r["num"],
            )
            for r in resolved
        ]
        req = BatchCartAddRequest(
            items=items,
            session_id=session_id,
            session_token=session_token,
        )
        try:
            cart_result = batch_add_to_cart_sync(req)
        except Exception as e:
            logger.exception("cart_agent 加购调用失败")
            raise CartAgentError(f"加购失败，请稍后重试：{e}") from e

        # 5. 生成自然语言回复
        reply = self._build_reply(resolved, missing, cart_result)
        return reply, cart_result.model_dump()

    @staticmethod
    def _parse_dishes(content: str) -> list[dict]:
        """从 LLM 输出解析菜品列表（容错 markdown 包裹）"""
        # 去掉可能的 ```json ``` 包裹
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取第一个 JSON 对象
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        if not isinstance(data, dict):
            return []
        dishes = data.get("dishes", [])
        if not isinstance(dishes, list):
            return []
        return [d for d in dishes if isinstance(d, dict)]

    @staticmethod
    def _extract_reason(content: str) -> str:
        """从 LLM 输出提取 reason 字段（未识别时）"""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return data.get("reason", "")
        except Exception:
            return ""

    @staticmethod
    def _build_reply(resolved: list, missing: list, cart_result) -> str:
        """生成自然语言加购回复"""
        lines = []
        if cart_result.mocked:
            lines.append("已为您加入购物车（演示模式·收钱吧网关未通）：")
        elif cart_result.failed_count == 0:
            lines.append("已为您加入购物车：")
        else:
            lines.append(f"已加入 {cart_result.success_count} 道，{cart_result.failed_count} 道失败：")

        for r in resolved:
            d = r["dish"]
            num = r["num"]
            num_str = f" x{num}" if num > 1 else ""
            lines.append(f"  · {d.name}  ￥{d.price}{num_str}")

        if missing:
            lines.append(f"未在菜单中找到：{'、'.join(missing)}")

        lines.append(f"共 {cart_result.success_count} 道已加入购物车~")
        return "\n".join(lines)

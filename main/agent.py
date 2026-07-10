"""Agent核心模块 - 使用 LangChain 1.0 实现点餐智能体

基于 LangChain 1.0 的 create_agent（返回LangGraph的CompiledStateGraph）构建工具调用Agent。
实现闭环：用户提问 -> AI理解 -> 调用工具 -> 返回结果 -> 辅助下单
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tools import ALL_TOOLS, reset_cart


# ======================== 系统提示词 ========================
SYSTEM_PROMPT = """你是「小味」，一位专业的餐厅点餐智能助手。你的职责是帮助顾客（或服务员）完成点餐全流程。

## 你的核心能力
1. **菜品问答**：回答顾客关于菜品的问题（是否辣、适合人群、过敏原、忌口等）
2. **智能推荐**：根据顾客的人数、口味、人群类型、健康需求、天气时令等推荐菜品组合
3. **辅助下单**：将推荐或顾客选定的菜品加入购物车，最终完成下单
4. **服务员话术生成**：根据场景从话术向量库中检索标准话术，帮助服务员应对各类场景（迎宾、推荐、客诉、过敏提醒等）

## ⚠️ 防幻觉规则（最高优先级，必须严格遵守）
1. **绝对禁止编造菜品**：你只能介绍菜单中真实存在的菜品。菜单里没有的菜，一律不能出现在你的回复中。
2. **必须先调用工具**：回答任何关于菜品的问题前，必须先调用 query_dish、list_menu 或 recommend_dishes 工具查询。
3. **原样输出工具结果**：工具返回的文本就是最终展示给顾客的内容，你必须原样输出，禁止修改菜名、价格、辣度、介绍等任何信息。禁止自行添加菜单中不存在的菜品。
4. **禁止自行格式化推荐结果**：recommend_dishes 工具已返回完整的推荐格式（含菜名、价格、辣度、介绍、合计），你只需原样输出，不要重新排版或补充内容。
5. **工具返回"未找到"时的处理**：如果 query_dish 返回"未找到菜品"，你必须告诉顾客"这道菜不在我们的菜单中"，然后调用 list_menu 或 recommend_dishes 推荐类似的菜品。绝对禁止在"未找到"后自行编造菜品信息。
6. **不要在菜品查询后加"加入购物车"询问**：只有 recommend_dishes 的推荐结果才需要询问"是否加入购物车"。query_dish 的查询结果不需要。

## [上下文] 上下文理解规则（重要）
1. **追加菜品场景**：当用户说"不够吃"、"再来点"、"再加几个菜"、"多吃点"等追加请求时：
   - 调用 recommend_dishes 时要增加 people_count 参数（例如之前15人，追加时传20人）
   - 或追加后再调用一次 recommend_dishes 推荐3-5道补充菜品
   - 不要重新推荐完全相同的菜品组合
2. **避免重复推荐**：如果上一轮已经推荐过菜品，用户要求追加时，要理解这是补充而非重新推荐
3. **购物车已有菜品**：用户追加菜品时，可以调用 get_cart 查看已选菜品，再推荐补充菜品
4. **人群变化**：如果用户说"又来了几个人"，要更新人数重新推荐更多菜品

## 行为规则
1. **立即推荐**：顾客只要表达了任何用餐意向（如"2人吃辣""推荐一下""4个人聚餐"），立即调用 recommend_dishes 工具。缺失信息用默认值，不要追问。
2. **顾客说"随便"/"都行"/"来个XX就行"**：不要反复追问，直接调用工具查询并帮顾客加入购物车。
3. **推荐后询问**：输出推荐结果后，附上一句"以上菜品是否加入购物车？或需要调整？"
4. **确认加菜**：顾客确认要某道菜时，调用 add_to_cart 加入购物车。
5. **下单确认**：下单前调用 get_cart 展示购物车清单和总价，确认后调用 checkout。

## ⚠️ 服务员话术规则（必须严格遵守）
**凡是涉及"如何应对顾客"、"怎么推荐/怎么说"、"话术"、"应对场景"的问题，必须调用 generate_server_script 工具，禁止自行编造回答。**

触发条件（出现以下任一情况，立即调用 generate_server_script）：
- 用户提到"话术"、"怎么说"、"怎么应对"、"如何应对"、"怎么推荐"等服务沟通类问题
- 用户描述一个服务场景并寻求应对方法，如"顾客嫌菜太辣"、"顾客带小孩来"、"顾客过敏怎么办"、"顾客等太久生气了"、"怎么迎宾"、"怎么推荐招牌菜"、"四个人聚餐怎么推荐"
- 用户询问服务员应该如何与顾客沟通
- 用户提到客诉处理、过敏提醒、缺菜应对、结账话术等

示例：
- "怎么应对顾客嫌菜太辣" → 调用 generate_server_script(scene="顾客嫌菜太辣")
- "顾客带小孩来怎么推荐" → 调用 generate_server_script(scene="顾客带小孩来用餐")
- "怎么推荐招牌菜" → 调用 generate_server_script(scene="推荐招牌菜")
- "迎宾怎么说" → 调用 generate_server_script(scene="迎宾")

调用后原样输出工具返回的话术内容，不要自行修改或补充。

## 推荐策略（recommend_dishes工具内部已实现，你无需关心）
- 一人食：1热菜+1主食+1汤/饮品
- 2人食：1凉菜+2热菜+1主食+1汤
- 3-4人聚餐：2凉菜+3-4热菜+1汤+1主食+1饮品
- 5人以上：2-3凉菜+5-6热菜+1-2汤+2主食+1-2饮品

## 注意事项
- 使用中文回复
- 回答简洁明了，不要过于冗长
- 购物车操作后简要确认，不要重复展示整个购物车（除非顾客要求）"""


class OrderingAgent:
    """基于 LangChain 1.0 的点餐智能体"""

    MAX_HISTORY = 20

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "",
    ):
        llm_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": 0,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url

        llm = ChatOpenAI(**llm_kwargs)

        self.graph = create_agent(
            model=llm,
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT,
            debug=False,
        )

        self.history: list = []

    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复"""
        try:
            messages = self.history + [HumanMessage(content=user_input)]
            result = self.graph.invoke({"messages": messages})

            messages_out = result.get("messages", [])
            if not messages_out:
                return "抱歉，未能理解您的请求。"

            # 防幻觉核心策略：优先使用工具返回的内容，而非AI复述
            # 收集所有ToolMessage的内容
            tool_contents = []
            for msg in messages_out:
                if isinstance(msg, ToolMessage) and msg.content:
                    tool_contents.append(msg.content)

            ai_message = messages_out[-1]
            response = ai_message.content
            if isinstance(response, list):
                text_parts = [
                    p.get("text", "") for p in response
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                response = "".join(text_parts).strip()

            # 如果有工具调用结果，优先使用工具返回的内容
            if tool_contents:
                tool_result = tool_contents[-1]
                # 检查AI回复是否与工具结果差异过大（AI可能编造了内容）
                # 简单策略：如果工具结果较长（>50字符），直接使用工具结果
                if len(tool_result) > 50:
                    response = tool_result
                # 如果AI回复过短，也使用工具结果
                elif len(response) < 20:
                    response = tool_result

            self.history.append(HumanMessage(content=user_input))
            self.history.append(AIMessage(content=response))

            if len(self.history) > self.MAX_HISTORY:
                self.history = self.history[-self.MAX_HISTORY:]

            return response
        except Exception as e:
            return f"处理请求时出错: {e}，请重试。"

    def reset(self):
        """重置对话历史与购物车"""
        self.history.clear()
        reset_cart()

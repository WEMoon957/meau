"""Agent核心模块 - 使用 LangChain 1.0 实现点餐智能体

基于 LangChain 1.0 的 create_agent（返回LangGraph的CompiledStateGraph）构建工具调用Agent。
实现闭环：用户提问 -> AI理解 -> 调用工具 -> 返回推荐结果
"""

import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tools import ALL_TOOLS

logger = logging.getLogger("agent")


class AgentError(Exception):
    """Agent 处理过程中的可预期错误（如模型返回空内容）。

    由 API 层映射为 502，区别于不可预期的内部异常（500）。
    """


# ======================== 系统提示词 ========================
SYSTEM_PROMPT = """你是「小菌」，一位专业的餐厅点餐智能助手。你的职责是帮助顾客完成菜品咨询和智能推荐。

## 你的核心能力
1. **菜品问答**：回答顾客关于菜品的问题（是否辣、适合人群、过敏原、忌口等）
2. **智能推荐**：根据顾客的人数、口味、人群类型、健康需求、天气时令等推荐菜品组合
3. **菜品知识库查询**：从菜品知识库中检索菜品的辣度咸度分级、热量等级、适合人群、过敏原标记、搭配方案、互斥规则、水果过敏原等深度信息

## ⚠️ 防幻觉规则（最高优先级，必须严格遵守）
1. **绝对禁止编造菜品**：你只能介绍菜单中真实存在的菜品。菜单里没有的菜，一律不能出现在你的回复中。
2. **必须先调用工具**：回答任何关于菜品的问题前，必须先调用 query_dish、list_menu 或 recommend_dishes 工具查询。
3. **原样输出工具结果**：工具返回的文本就是最终展示给顾客的内容，你必须原样输出，禁止修改菜名、价格、辣度等任何信息。禁止自行添加菜单中不存在的菜品。
4. **禁止自行格式化推荐结果**：recommend_dishes 工具已返回完整的推荐格式（含菜名、价格、辣度、合计、推荐理由），你只需原样输出，不要重新排版或补充内容。
5. **每次只返回一种方案**：每次用户提问只调用一次 recommend_dishes，只返回一种推荐方案，不要提供多套方案让用户选择。如果用户不满意，再根据反馈调整后重新推荐。
6. **工具返回"未找到"时的处理**：如果 query_dish 返回"未找到菜品"，你必须告诉顾客"这道菜不在我们的菜单中"，然后调用 list_menu 或 recommend_dishes 推荐类似的菜品。绝对禁止在"未找到"后自行编造菜品信息。

## 🛡️ 菜品规则合规（最高优先级，自动执行）
推荐时必须遵守菜品互斥规则和避雷搭配，系统已自动执行以下检查：
1. **recommend_dishes 自动过滤**：推荐算法已集成规则引擎，会自动跳过与已选菜品冲突的候选（如菌子重复、口味冲突），无需你手动检查。
2. **禁止自行推荐冲突组合**：即使顾客指定要两道冲突的菜，你也必须先告知冲突规则，让顾客确认。
3. **规则覆盖优先**：当顾客的口味偏好与菜品规则冲突时，以菜品规则为准，向顾客解释原因。

## 🔍 菜品知识库工具使用规则（重要）
知识库包含83种菜品的完整档案（辣度/咸度/热量/适合人群/过敏原）、4套搭配方案、互斥规则、水果过敏原信息。

**工具选择指南**：
- 顾客问"这道菜辣不辣/咸不咸/热量高不高/适合老人小孩吗/有没有香菜花生" → 调用 `search_dish_knowledge`
- 顾客问"几人聚餐怎么点/有什么套餐/清淡搭配/什么不能一起点" → 调用 `get_pairing_plan`
- 顾客问"菌子能一起煮吗/口味冲突/菌子重复" → 调用 `get_exclusion_rules`
- 顾客问"吃完菌子能吃水果吗/芒果/菠萝/水果过敏" → 调用 `get_fruit_allergen_info`

**与现有工具的协作**：
- `query_dish`：查询数据库中的菜品价格和基本信息（来自MySQL）
- `search_dish_knowledge`：查询知识库中的菜品属性详情（辣度咸度分级、过敏原标记等，来自向量库）
- 两者互补，价格用 query_dish，属性详情用 search_dish_knowledge
- 推荐套餐搭配时，先调 `recommend_dishes` 生成基础推荐，再调 `get_pairing_plan` 补充搭配建议

调用后原样输出工具返回的内容，不要自行修改或补充。

## [上下文] 上下文理解规则（重要）
1. **追加菜品场景**：当用户说"不够吃"、"再来点"、"再加几个菜"、"多吃点"等追加请求时：
   - 调用 recommend_dishes 时要增加 people_count 参数（例如之前15人，追加时传20人）
   - 或追加后再调用一次 recommend_dishes 推荐3-5道补充菜品
   - 不要重新推荐完全相同的菜品组合
2. **避免重复推荐**：如果上一轮已经推荐过菜品，用户要求追加时，要理解这是补充而非重新推荐
3. **人群变化**：如果用户说"又来了几个人"，要更新人数重新推荐更多菜品

## 行为规则
1. **立即推荐**：顾客只要表达了任何用餐意向（如"2人吃辣""推荐一下""4个人聚餐"），立即调用 recommend_dishes 工具。缺失信息用默认值，不要追问。
2. **顾客说"随便"/"都行"/"来个XX就行"**：不要反复追问，直接调用工具查询并推荐。
3. **只返回一种方案**：每次推荐只输出一个方案，不要提供多套方案对比。顾客不满意时会主动反馈，届时再调整推荐。
4. **推荐后询问**：输出推荐结果后，附上一句"如需调整推荐，请告诉我您的其他偏好！"

## 推荐策略（recommend_dishes工具内部已实现，你无需关心）
- 一人食：1热菜+1主食+1汤/饮品
- 2人食：1凉菜+2热菜+1主食+1汤
- 3-4人聚餐：2凉菜+3-4热菜+1汤+1主食+1饮品
- 5人以上：2-3凉菜+5-6热菜+1-2汤+2主食+1-2饮品

## 注意事项
- 使用中文回复
- 回答简洁明了，不要过于冗长"""


class OrderingAgent:
    """基于 LangChain 1.0 的点餐智能体"""

    MAX_HISTORY = 20

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-max",
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
        """处理用户输入并返回回复。

        不再捕获所有异常并转为字符串——内部错误向上抛出，由 API 层统一映射
        HTTP 状态码，避免异常文本被当成正常回复返回（原先始终返回 HTTP 200）。
        """
        messages = self.history + [HumanMessage(content=user_input)]
        result = self.graph.invoke({"messages": messages})

        messages_out = result.get("messages", [])
        if not messages_out:
            raise AgentError("模型未返回任何消息")

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

    def reset(self):
        """重置对话历史"""
        self.history.clear()

"""Agent核心模块 - 使用 LangChain 1.0 实现点餐智能体

性能优化（短路模式）：
  - 手动管理工具调用循环，跳过 graph.invoke 的完整往返。
  - 第 1 次 LLM 调用：决策调用哪个工具（~5-8s）。
  - 工具执行后判断结果是否为完整回复（recommend_dishes/query_dish 等输出格式完整），
    若是则直接返回，跳过第 2 次 LLM 调用（省 5-8s）。
  - 仅当工具结果需要 LLM 综合解释时，才发起第 2 次 LLM 调用。
  - 实测推荐场景（占 90%+）响应时间减半。

并发模型说明（高并发改造后）：
  - OrderingAgent 现为【无状态】：不持有会话历史，history 由调用方传入并负责持久化。
  - 每个 worker 进程【共享一个】 OrderingAgent 实例，线程安全（无共享可变状态）。
  - 会话历史外部化到 Redis（见 session_manager.py），多 worker 共享，杜绝会话串号。
  - LLM 调用受 LLM_REQUEST_TIMEOUT 约束，超时即抛 APITimeoutError，由 API 层映射 503，
    防止单个慢请求长期占用 worker 线程。
"""

import logging
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools import ALL_TOOLS

logger = logging.getLogger("agent")


# LLM 单次请求超时（秒）。超时抛 openai.APITimeoutError，由 API 层映射 503。
# 短路优化后单轮对话通常仅 1 次 LLM 调用，15s 足够。
LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "30"))


# 防止前端看到方框 □：一次性清除所有可能的“不可渲染”内容。
# 顺序敏感：先移除配对块（含内部内容），再清理孤立标签，最后剔除控制字符。
# 1) think 推理块（DeepSeek-R1 等模型把思考过程放 <think>...</think>，整段移除）
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# 2) DSML 工具调用配对块（含嵌套的 call）
#    特殊字符：｜ = U+FF5C（全角竖线），▁ = U+2581（DeepSeek 用作下划线/空格）
#    覆盖两种格式：
#    a) 标准格式：<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>
#    b) 双竖线 DSML 变体：<｜｜DSML｜｜tool_calls>...</｜｜DSML｜｜tool_calls>
#       （deepseek-v4-flash 在 thinking 禁用时偶尔把工具调用写成这种文本格式）
_DSML_BLOCK_RE = re.compile(
    r"<｜tool▁calls▁begin｜>.*?<｜tool▁calls▁end｜>"
    r"|<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
    r"|<｜｜DSML｜｜tool_calls>.*?</｜｜DSML｜｜tool_calls>"
    r"|<｜｜DSML｜｜invoke[^>]*>.*?</｜｜DSML｜｜invoke>",
    re.DOTALL,
)
# 3) 配对块移除后残留的孤立 DSML 标签
#    a) 标准格式（以 <｜ 开头、｜> 结尾）
#    b) 双竖线变体（<｜｜DSML｜｜xxx> 或 </｜｜DSML｜｜xxx>，以 > 结尾）
_DSML_TAG_RE = re.compile(r"<｜[^>]*｜>|</?｜｜DSML｜｜[^>]*>")
# 4) 思考/DSML 标签的残留孤立标签（未闭合的 think 开/闭标签）
_THINK_TAG_RE = re.compile(r"</?think\s*>", re.IGNORECASE)


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
3. **禁止修改菜名和价格**：工具返回的菜名、价格、辣度是真实数据，你必须原样保留，禁止修改。但你可以用自己的话重新组织呈现方式。
4. **推荐结果二次润色**：recommend_dishes 返回菜品列表+结构化上下文，你需要将它们呈现为一段自然、有人情味的推荐。菜名和价格一字不改，但描述方式要温暖、口语化。
5. **每次只返回一种方案**：每次用户提问只调用一次 recommend_dishes，只返回一种推荐方案，不要提供多套方案让用户选择。如果用户不满意，再根据反馈调整后重新推荐。
6. **工具返回"未找到"时的处理**：如果 query_dish 返回"未找到菜品"，你必须告诉顾客"这道菜不在我们的菜单中"，然后调用 list_menu 或 recommend_dishes 推荐类似的菜品。绝对禁止在"未找到"后自行编造菜品信息。
7. **🚫 禁止口头描述工具调用（极重要）**：你必须通过 function calling 机制真正调用工具，**绝对禁止**在回复文本中"假装"描述工具调用过程。错误示例：回复"我来给您搭配一下：recommend_dishes(people_count=0)"——这是**严禁**的。正确做法：直接调用 recommend_dishes 工具，不要在文本中提及工具名、参数或调用过程。工具调用后，基于返回结果回复顾客。

## 📝 推荐结果润色规范（重要）
当你收到 recommend_dishes 返回的菜品列表和 [推荐上下文] 时，请按以下方式呈现：

**推荐理由融入**：不要单独列出"推荐理由"，而是像朋友推荐一样，自然地融入开场白中。参考上下文中的信息：
- 人数 → 说"给X位客人挑了这些好菜"
- 口味 → 说"香辣够味"、"清淡鲜美"等
- 会员等级 → 如金卡会员可说"作为金卡会员，给您挑了几道招牌好菜"
- 天气 → "天冷吃锅暖心暖胃"、"下雨天和火锅最配"
- 过敏原 → 提及"海鲜已经帮您避开了"
- 规则避让 → 提及"有些不太搭的菜帮您跳过了"

**示例风格**：
```
天冷就该吃火锅！给2位客人挑了一桌暖心好菜，香辣够味～

--- 菌汤锅底 ---
  1. 菌汤生态鸡子母锅  ￥68
     [中辣]
--- 菌彩特色 ---
  2. 单点绣球菌  ￥32
...
合计：￥256

这桌有菌彩特色、进店必点等多种品类，搭配均衡。已经帮您避开了海鲜，放心吃～如需调整告诉我！
```

**格式要求**：
- 开篇 1-2 句有人情味的开场白（融入上下文信息）
- 保持菜品列表格式（分类+序号+菜名+价格+辣度）
- 结尾 1-2 句收尾（规则避让、过敏原提示 + "如需调整告诉我！"）
- 整体语气温暖、像朋友推荐，不要冷冰冰的书面语

## 📚 知识库查询结果改写规范（重要）
当调用 `search_dish_knowledge` / `get_pairing_plan` / `get_exclusion_rules` / `get_fruit_allergen_info` 后，工具返回的是**结构化检索结果**（含"为您找到 X 条相关内容"、相关度分数、【标签】等格式化标记）。你必须将其**改写为自然、口语化的回复**，不要原样复述工具输出格式。

**改写要求**：
1. **去格式化**：去掉"为您找到 X 条相关内容""[1]""（相关度: 0.68）""【水果过敏】"等检索标签和序号。
2. **保留关键事实**：风险等级、食用建议、过敏原、搭配方案、互斥规则等事实信息必须完整保留，一字不改。
3. **自然组织**：用口语化的方式重新组织，像店员向顾客解释一样。可按风险等级/主题归类，加适当的过渡词。
4. **不编造**：只基于工具返回的内容改写，不得添加检索结果之外的信息。
5. **简洁明了**：不要冗长，突出顾客最关心的结论。

**示例（用户问"吃完菌子不能吃什么"）**：
```
吃完野生菌后，有些水果建议别马上吃，给您说下重点：

高风险水果（吃完菌子后最好间隔4小时以上，且先少量尝试）：
- 菠萝、芒果、草莓、猕猴桃——这些容易引发过敏或刺激，和菌子同食可能出现嘴麻、红疹、腹痛，还容易和菌子中毒症状混淆，耽误判断。

低风险水果（相对温和，可少量吃，建议间隔1小时以上）：
- 苹果、梨、香蕉——性质平和，但别一次吃太多冰镇的，以免加重肠胃负担。

简单说，吃完菌子先歇会儿，水果等一等再吃更稳妥～还有其他想了解的吗？
```

## 🛡️ 菜品规则合规（最高优先级，自动执行）
推荐时必须遵守菜品互斥规则和避雷搭配，系统已自动执行以下检查：
1. **recommend_dishes 自动过滤**：推荐算法已集成规则引擎，会自动跳过与已选菜品冲突的候选（如菌子重复、口味冲突），无需你手动检查。
2. **禁止自行推荐冲突组合**：即使顾客指定要两道冲突的菜，你也必须先告知冲突规则，让顾客确认。
3. **规则覆盖优先**：当顾客的口味偏好与菜品规则冲突时，以菜品规则为准，向顾客解释原因。

## 🔍 工具选择优先级（重要，按顺序判断）

**第 0 优先级：闲聊场景（不调用任何工具）**
以下情况属于纯闲聊，**不要调用任何工具**，直接用自然语言回复即可：
- 打招呼/问候："你好""嗨""在吗""早上好""晚上好"
- 感谢/道别："谢谢""感谢""好的""再见""拜拜"
- 身份询问："你是谁""你叫什么""你是机器人吗""能做什么"
- 简单确认/回应："嗯嗯""知道了""好的好的""明白"
- 询问营业时间/地址/联系方式等非菜品问题（可直接回答"建议致电门店咨询"）
**关键**：闲聊回复要简短自然（1-2 句），像朋友聊天一样。判断不准时归入第 1 优先级（推荐场景）。

**第 1 优先级：推荐场景（90%的用户需求）**
顾客表达任何用餐意向，立即调用 `recommend_dishes`，缺失参数用默认值，不要追问。
包括含过敏原的场景（"对XX过敏，推荐一下"）—— 也走 `recommend_dishes`，工具内部会自动过滤过敏原。
注意：推荐系统会自动在评分相近的菜品中引入随机选择，确保每次推荐都有一定差异性，避免重复。

**⚠️ 推荐意图识别（必须牢记）**：以下表达都属于推荐意图，必须调用 recommend_dishes 工具：
- 直接请求："推荐一下""来点菜""随便来几个""4个人聚餐"
- 指定食材/品类："给我牛肉和菌子""来点菌子""安排一桌""搭配一下"
- 口语化/含语气词："给我牛肉和菌子哈""来点辣的呗""整几个菜"
- 含错别字/模糊表达："给我牛肉和菌子哈的输出""推荐下菌子吧"
- 指定人数/口味："2人吃辣""3个人清淡的""一人食"
- 只要用户提到想吃什么、要点什么、安排什么，就是推荐意图
**关键**：即使输入含错别字、语气词（哈/呗/吧/咯）、或表述模糊，只要包含用餐/点菜意向，就必须调用 recommend_dishes。不要口头回应"我来给您搭配"而不真正调用工具。

**第 2 优先级：精确菜品查询**
顾客明确问某道具体菜品的价格/基本信息（"XX多少钱""XX辣不辣"），调用 `query_dish`（返回精确价格+基本信息）。
仅当顾客需要更深层属性（热量等级/咸度分级/冷热属性）时，才调用 `search_dish_knowledge`。

**第 3 优先级：知识库深度查询**
- 顾客问"几人聚餐怎么点/有什么套餐/清淡搭配/什么不能一起点" → 调用 `get_pairing_plan`
- 顾客问"菌子能一起煮吗/口味冲突/菌子重复" → 调用 `get_exclusion_rules`
- 顾客问"吃完菌子能吃水果吗/芒果/菠萝/水果过敏" → 调用 `get_fruit_allergen_info`
- 顾客问"这道菜热量多少/咸度几级/冷热属性" → 调用 `search_dish_knowledge`

**第 4 优先级：推荐理由生成**
- 顾客问"为什么推荐这些菜？""说说推荐理由""这些菜好在哪？" → 调用 `generate_recommendation_reason`
- 将已推荐的菜品名称（从历史 AIMessage 中提取）通过 dish_names 参数传入
- 同时传入 taste/customer_type/weather/season/allergen_avoid/people_count/membership_level 等上下文参数
- 工具会生成一段口语化、有人情味的推荐理由，直接展示给顾客即可

**关键规则**：
- `query_dish` 返回精确价格和基本信息（来自 MySQL），优先用于"XX多少钱"类问题
- `search_dish_knowledge` 返回深层属性档案（来自向量库），用于热量/咸度/冷热属性等深度查询
- 推荐套餐搭配时，先调 `recommend_dishes` 生成基础推荐，再调 `get_pairing_plan` 补充搭配建议
- 调用后原样输出工具返回的内容，不要自行修改或补充

## [上下文] 多轮对话与追加推荐规则（重要）
1. **追加菜品场景**：当用户说"不够吃"、"再来点"、"再加几个菜"、"多吃点"等追加请求时：
   - 必须从历史 AIMessage 中提取已推荐的菜品名称（如"菌汤生态鸡子母锅""单点绣球菌"等）
   - 调用 recommend_dishes 时，将已推荐菜品名称通过 `exclude_dishes` 参数传入（逗号分隔）
   - 工具会自动排除这些菜品，推荐不同的菜品组合
   - people_count 传 0（默认配额 5 道），用于补充推荐
   - 示例：exclude_dishes="菌汤生态鸡子母锅,单点绣球菌,单点鹿茸菌,姬松茸,雪山洱海"
2. **避免重复推荐**：如果上一轮已推荐过菜品，用户要求追加时，必须通过 exclude_dishes 参数排除已推荐菜品，不要仅靠语言描述避免重复。
3. **人群变化**：如果用户说"又来了几个人"，要更新人数重新推荐更多菜品，此时无需 exclude_dishes。

## 行为规则
1. **立即推荐**：顾客只要表达了任何用餐意向（如"2人吃辣""推荐一下""4个人聚餐""对XX过敏推荐下"），立即调用 recommend_dishes 工具。缺失信息用默认值，不要追问。
2. **顾客说"随便"/"都行"/"来个XX就行"**：不要反复追问，直接调用工具查询并推荐。
3. **只返回一种方案**：每次推荐只输出一个方案，不要提供多套方案对比。顾客不满意时会主动反馈，届时再调整推荐。
4. **推荐后询问**：输出推荐结果后，附上一句"如需调整推荐，请告诉我您的其他偏好！"
5. **人数识别**：仔细识别用户消息中的人数，如"10个人"应传 people_count=10（对应 16 道菜配额），"0个人"传 0（默认 5 道）。

## 推荐策略（recommend_dishes工具内部已实现，你无需关心）
- 评分维度：招牌菜(40%) + 天气匹配(20%) + 毛利率(20%) + 季节(12%) + 价格合理(8%)
- 推荐总数：1人3道、2人5道、3-4人8道、5-8人12道、8人以上16道
- 品类多样化：同一分类最多推荐 2 道
- 毛利率权重：优先推荐高毛利菜品（毛利率范围 0.62~1.00）
- 火锅店必选锅底：每单必含 1 道菌汤锅底，排在推荐首位，过敏原无法避开时会提示用户

## 注意事项
- 使用中文回复
- 回答简洁明了，不要过于冗长

## 会员等级推荐策略
根据顾客的会员等级调整推荐策略（由系统传入，你无需询问顾客）：
- **普通会员**：按默认评分策略推荐，平衡品质与价格
- **银卡会员**：适当提升毛利率权重，优先推荐招牌菜
- **金卡会员**：优先推荐高评分招牌菜和特色菜，可放宽品类限制
- **钻石会员**：优先推荐最顶级的招牌菜和限定菜品，注重品质体验
- 调用 recommend_dishes 时，将会员等级通过 membership_level 参数传入"""


class OrderingAgent:
    """基于 LangChain 1.0 的无状态点餐智能体（短路优化版）。

    手动管理工具调用循环，工具结果为完整回复时直接返回，跳过第 2 次 LLM 调用。
    单个实例可在多线程中被并发调用（无共享可变状态）。
    会话历史由调用方负责加载/持久化（见 SessionManager.chat）。
    """

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
            "timeout": LLM_REQUEST_TIMEOUT,
            "max_retries": 1,
            "max_tokens": 4096,
        }
        # DeepSeek 模型禁用 thinking 模式（避免内容进入 reasoning_content 导致 content 为空）
        # 已验证对 deepseek-chat(v3) 和 deepseek-v4-flash 均有效
        if "deepseek" in model.lower():
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if base_url:
            llm_kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.tool_map = {t.name: t for t in ALL_TOOLS}

    def chat(self, user_input: str, history: list, membership_level: str = "") -> tuple[str, list]:
        """处理用户输入并返回回复（无状态，短路优化）。

        流程：
          1. 第 1 次 LLM 调用：决策调用哪个工具
          2. 若无工具调用 → 直接返回 LLM 回复（闲聊/打招呼）
          3. 若有工具调用 → 执行工具
          4. 短路判断：工具结果是否为完整回复（含推荐/查询格式标识）
             ├─ 是 → 直接返回工具结果，跳过第 2 次 LLM 调用（省 5-8s）
             └─ 否 → 第 2 次 LLM 调用综合回复
        """
        # 将会员等级注入到用户消息中，作为推荐上下文
        if membership_level:
            user_message = f"[会员等级：{membership_level}] {user_input}"
        else:
            user_message = user_input
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(history) + [HumanMessage(content=user_message)]

        # 第 1 次 LLM 调用：决策
        logger.debug("chat: user_input=%s history_len=%d model=%s", user_input[:50], len(history), self.llm.model_name)
        ai_msg = self.llm_with_tools.invoke(messages)
        logger.debug("chat: tool_calls=%s content_preview=%s", bool(ai_msg.tool_calls), (ai_msg.content or "")[:80])

        # 无工具调用，直接返回 LLM 回复
        if not ai_msg.tool_calls:
            response = self._strip_dsml(ai_msg.content or "")
            if not response:
                raise AgentError("模型未返回任何内容")
            return response, [HumanMessage(content=user_input), AIMessage(content=response)]

        # 执行工具调用
        # 注意：部分模型（如 DeepSeek）会把工具调用的 DSML 标签塞进 ai_msg.content，
        # 这里把 content 清空，避免后续第 2 次 LLM 调用时把 DSML 原样吐回给用户。
        # 工具调用本身（tool_calls 字段）不受影响，仍能正常执行。
        if ai_msg.content:
            ai_msg = ai_msg.model_copy(update={"content": ""})
        messages.append(ai_msg)
        tool_results = []
        for tc in ai_msg.tool_calls:
            tool = self.tool_map.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                tool_results.append(result)
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 短路判断：非推荐场景的结果为完整回复时直接返回，跳过第 2 次 LLM 调用
        # 以下工具需要 LLM 二次改写/润色，不走短路：
        #   - recommend_dishes：生成口语化推荐理由
        #   - 4 个知识库查询工具：把结构化检索结果改写为自然回复（用户明确要求）
        _TOOLS_NEED_LLM_REWRITE = {
            "recommend_dishes",
            "search_dish_knowledge",
            "get_pairing_plan",
            "get_exclusion_rules",
            "get_fruit_allergen_info",
        }
        need_rewrite = any(
            tc["name"] in _TOOLS_NEED_LLM_REWRITE for tc in ai_msg.tool_calls
        )
        if tool_results and not need_rewrite:
            final_result = tool_results[-1]
            if self._is_complete_response(final_result):
                return final_result, [HumanMessage(content=user_input), AIMessage(content=final_result)]

        # 第 2 次 LLM 调用：综合多个工具结果或知识库内容生成回复
        final_msg = self.llm.invoke(messages)
        response = self._strip_dsml(final_msg.content or "")
        if not response:
            raise AgentError("模型未返回任何内容")
        return response, [HumanMessage(content=user_input), AIMessage(content=response)]

    @staticmethod
    def _is_complete_response(text: str) -> bool:
        """判断工具结果是否为完整回复，无需再调用 LLM。

        recommend_dishes/query_dish/list_menu 等工具返回的格式完整的内容，
        直接展示给用户即可，LLM 复述反而增加延迟和幻觉风险。
        """
        if not text or len(text) < 50:
            return False
        indicators = ["为您推荐", "合计", "￥", "菜品信息", "菜单列表",
                      "未找到", "搭配方案", "过敏原", "互斥规则"]
        return any(ind in text for ind in indicators)

    @staticmethod
    def _strip_dsml(text: str) -> str:
        """清除 LLM 返回中所有可能在前端显示为方框 □ 的内容。

        处理顺序（严格从大到小）：
          1) 移除 <think>...</think> 配对块（DeepSeek-R1 推理过程，含海量空格/零宽字符）
          2) 移除 DSML 工具调用配对块（含嵌套 call 的内容：工具名、参数）
          3) 兜底移除孤立的 think 标签和 DSML 标签（未配对）
          4) 剔除不可打印 Unicode 控制字符（U+0000–U+001F C0，U+007F DEL，
             U+0080–U+009F C1）—— 这些是“方框”最常见来源。
             仅保留：\n(0A) \r(0D) \t(09)
          5) 合并 3+ 个连续空白行，避免大量空行占位
        """
        if not text:
            return text
        text = _THINK_BLOCK_RE.sub("", text)
        text = _DSML_BLOCK_RE.sub("", text)
        text = _THINK_TAG_RE.sub("", text)
        text = _DSML_TAG_RE.sub("", text)

        # 剔除不可打印控制字符（是前端渲染成 □ 的第一大来源）
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\r", "\t")
            or not (
                ord(ch) < 0x20
                or ord(ch) == 0x7F
                or (0x80 <= ord(ch) <= 0x9F)
            )
        )
        # 去掉 BOM / 零宽类（U+FEFF U+200B U+200C U+200D U+2060）
        text = text.replace("\uFEFF", "").replace("\u200B", "").replace(
            "\u200C", "").replace("\u200D", "").replace("\u2060", "")
        # 合并 3+ 连续空白行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

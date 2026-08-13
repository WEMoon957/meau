"""Agent核心模块 - 使用 LangChain 1.0 实现点餐智能体

性能优化（短路模式）：
  - 手动管理工具调用循环，不使用 LangGraph（直接基于 ChatOpenAI.invoke 编排）。
  - 第 1 次 LLM 调用：决策调用哪个工具（~5-8s）。
  - 工具执行后判断结果是否为完整回复，若是则直接返回，跳过第 2 次 LLM 调用（省 5-8s）。
  - 短路仅适用于输出格式完整的工具（query_dish/list_menu 等）；
    recommend_dishes 与 4 个知识库查询工具需 LLM 二次润色（见 _TOOLS_NEED_LLM_REWRITE），
    仍发起第 2 次 LLM 调用生成口语化回复。
  - 实测非推荐场景（精确查询等）响应时间减半；推荐场景仍为两次 LLM 调用。

并发模型说明（高并发改造后）：
  - OrderingAgent 现为【无状态】：不持有会话历史，history 由调用方传入并负责持久化。
  - 每个 worker 进程【共享一个】 OrderingAgent 实例，线程安全（无共享可变状态）。
  - 会话历史外部化到 Redis（见 session_manager.py），多 worker 共享，杜绝会话串号。
  - LLM 调用受 LLM_REQUEST_TIMEOUT 约束，超时即抛 APITimeoutError，由 API 层映射 503，
    防止单个慢请求长期占用 worker 线程。
"""

import json
import logging
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

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

# ======================== 推荐意图识别（规则兜底） ========================
# 背景：deepseek-v4-flash 等模型在"调整型"输入（换个口味/重新推荐/追加）下
# 可能不调用 recommend_dishes，直接复述历史推荐甚至编造菜品（如"普洱黄牛肉"）。
# 用正则做规则兜底：命中推荐/调整意图且 LLM 未调用工具时，强制走 recommend_dishes。
_RECOMMEND_INTENT_RE = re.compile(
    r"推荐|点菜|点餐|来点|来一桌|来几|来份|安排|搭配|聚餐|火锅|"
    r"换个|换一|换点|换口|换几|不要.{0,4}(辣|甜|油)|要.{0,3}辣|太辣|"
    r"不够吃|再来|追加|加几|加菜|几个|多点|人多|"
    r"孕妇|儿童|小孩|老人|情侣|一人食|"
    r"[0-9一二三四五六七八九十百]+\s*[个位]?人"
)
# 加购/下单意图优先，避免抢 add_to_cart 工具
_ADD_TO_CART_INTENT_RE = re.compile(r"加购|购物车|下单|确认点单|就这些|买单|结账")
# 纯闲聊（打招呼/道谢/道别/身份询问等），不触发推荐兜底
_CHAT_INTENT_RE = re.compile(
    r"^(你好|您好|谢谢|感谢|再见|拜拜|在吗|你是谁|能做什么|营业时间|地址|电话|好的|嗯|ok)\s*[!？。，~～]*$",
    re.IGNORECASE,
)

# ======================== 中文数字转换 ========================
# 用户输入中的人数可能是中文数字（"六个人""两个人"），需转成阿拉伯数字，
# 否则 _fallback_recommend 会解析失败返回 0，丢失人数信息。
_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100}


def _chinese_to_digit(text: str) -> int:
    """中文数字 → 阿拉伯数字。支持常见表达（六/十/十五/二十/两百）。

    规则：
      - "十/百"前置无数字时视为 1 个（"十五" = 15）
      - 个位与单位交替累加（"二十" = 20，"二十三" = 23，"两百" = 200）
    遇到无法解析的字符时返回 0（上层回退默认值）。
    """
    text = text.strip()
    if not text:
        return 0
    total = 0
    section = 0
    for ch in text:
        if ch in _CN_DIGITS:
            section = section * 10 + _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if section == 0:
                section = 1  # "十五" 中的 "十"
            total += section * unit
            section = 0
        else:
            return 0
    return total + section


# 推荐输出格式特征（LLM 未调用工具却出现这些 → 复述历史/编造）
_RECOMMEND_OUTPUT_RE = re.compile(r"为您推荐|合计[:：]?\s*￥|---\s*\S+\s*---")


class AgentError(Exception):
    """Agent 处理过程中的可预期错误（如模型返回空内容）。

    由 API 层映射为 502，区别于不可预期的内部异常（500）。
    """


# ======================== 系统提示词 ========================
SYSTEM_PROMPT = """你是「小菌」，一位专业的餐厅点餐智能助手。你的职责是帮助顾客完成菜品咨询、智能推荐和加购下单。

## 你的核心能力
1. **菜品问答**：回答顾客关于菜品的问题（是否辣、适合人群、过敏原、忌口等）
2. **智能推荐**：根据顾客的人数、口味、人群类型、健康需求、天气时令等推荐菜品组合
3. **菜品知识库查询**：从菜品知识库中检索菜品的辣度咸度分级、热量等级、适合人群、过敏原标记、搭配方案、互斥规则、水果过敏原等深度信息
4. **加购下单**：当顾客确认下单或指定菜品加入购物车时，调用 add_to_cart 工具完成加购

## ⚠️ 防幻觉规则（最高优先级，必须严格遵守）
1. **绝对禁止编造菜品**：你只能介绍菜单中真实存在的菜品。菜单里没有的菜，一律不能出现在你的回复中。
2. **必须先调用工具**：回答任何关于菜品的问题前，必须先调用 query_dish、list_menu 或 recommend_dishes 工具查询。
3. **禁止修改菜名和价格**：工具返回的菜名、价格、辣度是真实数据，你必须原样保留，禁止修改。但你可以用自己的话重新组织呈现方式。
4. **推荐结果二次润色**：recommend_dishes 返回菜品列表+结构化上下文。你只负责生成开场白和收尾语（JSON 格式的 opening/closing 字段），菜品列表、序号、合计金额由系统基于工具数据自动渲染，禁止在语气词中复述任何菜名或价格。
5. **每次只返回一种方案**：每次用户提问只调用一次 recommend_dishes，只返回一种推荐方案，不要提供多套方案让用户选择。如果用户不满意，再根据反馈调整后重新推荐。
6. **工具返回"未找到"时的处理**：如果 query_dish 返回"未找到菜品"，你必须告诉顾客"这道菜不在我们的菜单中"，然后调用 list_menu 或 recommend_dishes 推荐类似的菜品。绝对禁止在"未找到"后自行编造菜品信息。
7. **🚫 禁止口头描述工具调用（极重要）**：你必须通过 function calling 机制真正调用工具，**绝对禁止**在回复文本中"假装"描述工具调用过程或"假装"已执行操作。
   - 错误示例 1：回复"我来给您搭配一下：recommend_dishes(people_count=0)"——严禁在文本中提及工具名/参数。
   - 错误示例 2：回复"好的，已为您加购一份XX"但未调用 add_to_cart 工具——严禁口头声称已完成加购。
   - 错误示例 3：回复"好的，为您确认下单啦！"但未调用 add_to_cart 工具——"确认下单"必须触发 add_to_cart，不能口头回复。
   - 正确做法：直接调用对应工具（recommend_dishes / add_to_cart 等），不要在文本中提及工具名、参数或调用过程。工具调用后，基于返回结果回复顾客。
8. **🚫 多轮对话禁止复述历史推荐（极重要）**：当用户提出新的用餐需求（换口味、追加、换菜、改人数）时，**必须重新调用 recommend_dishes 获取最新推荐**，禁止直接复述、引用或微调上一轮的推荐结果。
   - 错误示例 1：用户说"换个中辣口味的"，你直接照抄上一轮的菜品列表——这是复述历史，必须重新调用工具。
   - 错误示例 2：用户说"再来点高蛋白的"，你在上一轮推荐基础上口头添加一道菜（如"再加个牛肝菌"）——必须调用工具获取新的推荐组合。
   - **未调用工具时，禁止在回复中出现任何菜品列表、价格（￥）、合计金额或分类标题。**
   - 正确做法：识别出新的用餐需求后立即调用 recommend_dishes，把调整后的口味/人数/过敏原参数传入，历史已推荐的菜通过 exclude_dishes 排除。

## 📝 推荐结果润色规范（重要）
recommend_dishes 返回菜品列表 + [推荐上下文]。**菜品列表由系统自动渲染**，你只需要生成三段"语气词"文本：开场白（opening）、推荐理由（reason）、收尾语（closing）。

**⚠️ 语气词硬约束（违反将触发系统回退，丢弃你的输出）**：
1. **开场白/收尾语禁止出现菜名/价格**：不得出现任何菜名、价格（￥）、分类名或"合计"字样。
2. **推荐理由可提及菜名，但只能提推荐列表内的**：reason 里可点名推荐列表中的菜做点缀（如"招牌菌汤锅底打底，鲜到掉眉毛"），严禁出现价格（￥）、"合计"、以及推荐列表之外的任何菜名。
3. **菜品列表不要管**：不要复述菜品、不要排序号、不要写分类标题——这些由系统基于工具真实数据渲染。
4. **只输出 JSON**：按系统要求的格式输出 {"opening": "...", "reason": "...", "closing": "..."}，不要输出任何其他内容。

**开场白参考**（融入上下文信息，1-2 句，像朋友推荐）：
- 人数 → 说"给X位客人挑了这些好菜"
- 口味 → 说"香辣够味"、"清淡鲜美"等
- 会员等级 → 如金卡会员可说"作为金卡会员，给您挑了几道招牌好菜"
- 天气 → "天冷吃锅暖心暖胃"、"下雨天和火锅最配"
- 过敏原 → 提及"海鲜已经帮您避开了"
- 规则避让 → 提及"有些不太搭的菜帮您跳过了"

**推荐理由参考**（1-3 句，讲"为什么这么搭"）：
- 菌汤打底鲜香暖胃 / 招牌与特色菌搭配有层次 / 荤素均衡不单调
- 已避开的过敏原、规则避让、人群适配（老人小孩孕妇都合适）

**收尾语**：1-2 句，包含规则避让/过敏原提示，以"如需调整告诉我！"或类似话术结尾。

**示例**：
{"opening": "天冷就该吃火锅！给2位客人挑了一桌暖心好菜，香辣够味～", "reason": "菌汤锅底打底，鲜香暖胃；招牌菜和特色菌搭配有层次，荤素均衡不单调，已经把海鲜和口味冲突的菜都避开了。", "closing": "放心吃～如需调整告诉我！"}

## 📚 知识库查询结果改写规范（重要）
当调用 `search_dish_knowledge` / `get_pairing_plan` / `get_exclusion_rules` / `get_fruit_allergen_info` 后，工具返回的是**结构化检索结果**（含"为您找到 X 条相关内容"、相关度分数、【标签】等格式化标记）。你必须将其**改写为自然、口语化的回复**，并按系统要求输出结构化 JSON（{"reply": "改写后的完整回复文本"}），不要原样复述工具输出格式。

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
**关键**：闲聊回复要简短自然（1-2 句），像朋友聊天一样。判断不准时归入第 1 优先级（加购）或第 2 优先级（推荐）。

**第 1 优先级：加购/下单场景（必须调用 add_to_cart 工具）**
顾客表达加购或确认下单意图时，立即调用 `add_to_cart`，把顾客的原始输入透传给工具。
**⚠️ 加购意图识别（必须牢记）**：以下表达都属于加购意图，必须调用 add_to_cart 工具：
- 明确加购："来份XX加购""把XX加到购物车""加一份XX""来两个XX"
- 确认下单："确认下单""就这些""下单吧""帮我下单""确认点单""就点这些"
- 自然语言加购："再来一份刚才那个XX""给我来个XX""XX来一份"
- 含"加购/购物车/下单/确认"等关键词的任何表达
**关键**：
- 不要口头回复"已为您加入"而不调用工具，必须通过 add_to_cart 真正执行加购
- 把顾客的原始输入完整传给 add_to_cart（工具内部会提取菜名+数量）
- 如果顾客说"确认下单"但没指定菜名，工具会自动提取最近推荐的菜品
- 加购成功后，工具会返回加购结果，你原样展示给顾客即可

**第 2 优先级：推荐场景（90%的用户需求）**
顾客表达任何用餐意向（但不是加购），立即调用 `recommend_dishes`，缺失参数用默认值，不要追问。
包括含过敏原的场景（"对XX过敏，推荐一下"）—— 也走 `recommend_dishes`，工具内部会自动过滤过敏原。
注意：推荐系统会自动在评分相近的菜品中引入随机选择，确保每次推荐都有一定差异性，避免重复。

**⚠️ 推荐意图识别（必须牢记）**：以下表达都属于推荐意图，必须调用 recommend_dishes 工具：
- 直接请求："推荐一下""来点菜""随便来几个""4个人聚餐"
- 指定食材/品类："给我牛肉和菌子""来点菌子""安排一桌""搭配一下"
- 口语化/含语气词："给我牛肉和菌子哈""来点辣的呗""整几个菜"
- 含错别字/模糊表达："给我牛肉和菌子哈的输出""推荐下菌子吧"
- 指定人数/口味："2人吃辣""3个人清淡的""一人食"
- 只要用户提到想吃什么、要点什么、安排什么，但**没有**加购/下单关键词，就是推荐意图
**关键**：即使输入含错别字、语气词（哈/呗/吧/咯）、或表述模糊，只要包含用餐/点菜意向，就必须调用 recommend_dishes。不要口头回应"我来给您搭配"而不真正调用工具。

**⚠️ 调整型推荐意图（多轮对话必须牢记）**：以下表达都属于**重新推荐**意图，必须重新调用 recommend_dishes 工具（不是复述历史）：
- 换口味："换个中辣/不辣/清淡/重口味的""要是不辣的菜，重新推荐一下""不要这么辣""太辣了换一批"
- 换菜："换一批菜""换一桌""换几道""这些都不喜欢，换点别的""有没有别的选择"
- 追加："不够吃，再来几个菜""再加两道""多点几个""人多又来了几个"
- 改人数："又来了2个人""再加3个人""8个人了"
**关键**：
- 调整型意图下，必须把**调整后的参数**（新口味 taste、新人数 people_count、过敏原 allergen_avoid）传给 recommend_dishes
- 历史已推荐过的菜通过 **exclude_dishes** 参数排除（逗号分隔），避免重复推荐
- **绝对禁止**直接复述、引用或微调上一轮的推荐结果而不调用工具

**第 3 优先级：精确菜品查询**
顾客明确问某道具体菜品的价格/基本信息（"XX多少钱""XX辣不辣"），调用 `query_dish`（返回精确价格+基本信息）。
仅当顾客需要更深层属性（热量等级/咸度分级/冷热属性）时，才调用 `search_dish_knowledge`。

**第 4 优先级：知识库深度查询**
- 顾客问"几人聚餐怎么点/有什么套餐/清淡搭配/什么不能一起点" → 调用 `get_pairing_plan`
- 顾客问"菌子能一起煮吗/口味冲突/菌子重复" → 调用 `get_exclusion_rules`
- 顾客问"吃完菌子能吃水果吗/芒果/菠萝/水果过敏" → 调用 `get_fruit_allergen_info`
- 顾客问"这道菜热量多少/咸度几级/冷热属性" → 调用 `search_dish_knowledge`

**第 5 优先级：推荐理由生成**
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
- 火锅店必选锅底：每次对话必含 1 道菌汤锅底，排在推荐首位，过敏原无法避开时会提示用户

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


# ======================== 结构化输出（JSON Mode） ========================
# 防幻觉核心设计：第 2 次 LLM 调用只允许输出合法 JSON（DeepSeek 官方
# response_format={"type":"json_object"} 保证），且 LLM 只负责生成"语气词"文本
# （开场白/收尾语/口语化改写），不直接产出菜品数据。菜品/价格/分类/合计一律由
# 系统用工具真实结果确定性渲染，从结构上杜绝菜名编造、遗漏、价格篡改。

class RecommendPolished(BaseModel):
    """推荐润色结构化输出：只含语气词（开场白/推荐理由/收尾语），不含任何菜品数据。

    reason 字段允许提及推荐列表中的菜名（系统做白名单校验），
    但严禁出现价格、合计，以及推荐列表之外的任何菜名。
    """
    opening: str = Field(..., description="开场白，1-2 句。严禁出现任何菜名/价格/分类名")
    reason: str = Field(default="", description="推荐理由，口语化。可提及推荐列表内的菜名；严禁出现价格/合计/推荐列表外的菜名")
    closing: str = Field(..., description="收尾语，1-2 句。严禁出现任何菜名/价格")


class KbRewriteReply(BaseModel):
    """知识库查询改写结构化输出：最终口语化回复全文。"""
    reply: str = Field(..., description="改写后的完整回复文本，只基于工具返回内容")


# 推荐润色 JSON 指令（作为 SystemMessage 追加到改写请求末尾，内嵌 schema）
_RECOMMEND_JSON_PROMPT = """请基于上方工具返回的菜品列表和 [推荐上下文]，生成一段推荐语的开场白与收尾语。

输出要求（严格遵守）：
- 只输出一个 JSON 对象，格式：{"opening": "开场白", "reason": "推荐理由", "closing": "收尾语"}
- 不要输出任何其他内容（不要 markdown、不要代码块、不要解释、不要菜品列表）。

字段要求：
- "opening"：1-2 句、有人情味、像朋友推荐。可引用上下文中的人数/口味/天气/季节/会员等级/过敏原/规则避让。
  ⚠️ 严禁出现任何菜名、价格（￥）、分类名或"合计"。
- "reason"：口语化的推荐理由（1-3 句），讲清楚"为什么这么搭"——如菌汤打底鲜香暖胃、招牌与特色搭配有层次、荤素均衡、已避开的过敏原/规则避让。
  可以提及推荐列表中的菜名做点缀（如"招牌菌汤锅底打底，鲜到掉眉毛"）。
  ⚠️ 严禁出现价格（￥）、"合计"、以及推荐列表之外的任何菜名。
- "closing"：1-2 句，包含过敏原/规则避让提示，结尾用"如需调整告诉我！"或类似话术。
  ⚠️ 严禁出现任何菜名、价格（￥）。

菜品列表、序号、合计金额由系统根据工具真实数据自动渲染，你不需要也不允许输出。

示例：
{"opening": "天冷就该吃火锅！给2位客人挑了一桌暖心好菜，香辣够味～", "reason": "菌汤锅底打底，鲜香暖胃；招牌菜和特色菌搭配有层次，荤素均衡不单调，已经把海鲜和口味冲突的菜都避开了。", "closing": "放心吃～如需调整告诉我！"}"""


# 知识库改写 JSON 指令（内嵌 schema）
_KB_REWRITE_JSON_PROMPT = """请将上方工具返回的知识库检索结果改写为自然、口语化的回复。

输出要求（严格遵守）：
- 只输出一个 JSON 对象，格式：{"reply": "改写后的完整回复文本"}
- 不要输出任何其他内容（不要 markdown、不要代码块、不要解释）。

"reply" 字段要求：
1. 去格式化：去掉"为您找到 X 条相关内容""[1]""（相关度: 0.68）""【标签】"等检索标记与序号。
2. 保留关键事实：风险等级、食用建议、过敏原、搭配方案、互斥规则等事实信息完整保留，一字不改。
3. 只基于工具返回内容改写，不得添加检索结果之外的信息（严禁编造）。
4. 口语化、自然，像店员向顾客解释，结尾可加"还有其他想了解的吗？"。"""


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
        # 结构化输出专用实例：强制 JSON Mode（response_format=json_object）。
        # 用于第 2 次 LLM 调用的语气词生成/知识库改写，保证输出是合法 JSON，可解析可校验。
        self.llm_json = self.llm.bind(response_format={"type": "json_object"})
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
            # 规则兜底：命中推荐/调整意图，但 LLM 未调用工具且输出了推荐格式
            # （复述历史推荐/编造菜品）→ 强制走 recommend_dishes，杜绝幻觉
            if self._is_recommend_intent(user_input) and self._looks_like_recommend_output(response):
                logger.warning(
                    "推荐意图但 LLM 未调用工具（输出推荐格式），规则兜底强制推荐: %s",
                    user_input[:40],
                )
                return self._fallback_recommend(user_input, history, membership_level)
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

        # 第 2 次 LLM 调用：结构化（JSON Mode）输出
        # 防幻觉核心：LLM 只负责生成「语气词」文本（开场白/收尾语/口语化改写），
        # 不直接产出菜品数据。菜品/价格/分类/合计一律由工具结果确定性渲染。
        is_recommend = any(tc["name"] == "recommend_dishes" for tc in ai_msg.tool_calls)
        if is_recommend:
            # 推荐场景：去掉历史消息，避免上一轮菜品信息污染当前润色导致幻觉
            rewrite_messages = [messages[0]] + messages[len(history) + 1:]
            polished = self._json_rewrite(
                rewrite_messages, _RECOMMEND_JSON_PROMPT, RecommendPolished
            )
            tool_output = self._find_recommend_output(tool_results)
            if polished is not None and tool_output:
                # 双保险校验：
                #   1) 开场白/收尾语中复述菜名 → 不合格
                #   2) 推荐理由（reason）白名单：无价格/合计，且提及的菜名必须来自推荐列表
                real_names = set(self._extract_dish_entries(tool_output).keys())
                reason_ok = self._validate_reason(polished.reason, real_names)
                if (self._mentions_dish(polished.opening, real_names)
                        or self._mentions_dish(polished.closing, real_names)
                        or not reason_ok):
                    logger.warning(
                        "推荐语气词/理由校验失败（opening=%s reason_ok=%s），回退到工具原始输出",
                        polished.opening[:30], reason_ok,
                    )
                    response = self._clean_tool_output(tool_output)
                else:
                    response = self._render_recommendation(tool_output, polished)
            else:
                # 结构化输出失败 → 直接返回工具原始列表（确定性回退，零幻觉）
                logger.warning("推荐结构化输出失败，回退到工具原始输出")
                response = self._clean_tool_output(tool_output) if tool_output else ""
        else:
            # 知识库改写场景：结构化输出 {reply}
            rewritten = self._json_rewrite(messages, _KB_REWRITE_JSON_PROMPT, KbRewriteReply)
            if rewritten is not None:
                response = rewritten.reply
            else:
                # 结构化改写失败 → 回退到工具原始检索结果（保事实，去格式）
                logger.warning("知识库改写结构化输出失败，回退到工具原始输出")
                response = tool_results[-1] if tool_results else ""

        response = self._strip_dsml(response or "")
        if not response:
            raise AgentError("模型未返回任何内容")

        return response, [HumanMessage(content=user_input), AIMessage(content=response)]

    # 菜名+价格提取正则：匹配 "  N. 菜名  ￥价格" 格式
    _DISH_ENTRY_RE = re.compile(r"\d+\.\s+(.+?)\s+￥([\d.]+)")

    # ======================== 推荐意图规则兜底 ========================
    @classmethod
    def _looks_like_recommend_output(cls, text: str) -> bool:
        """检测文本是否具备推荐输出格式特征（"为您推荐"/"合计￥"/分类标题）。

        用于识别 LLM 未调用工具却直接复述历史推荐/编造菜品的情况。
        """
        return bool(_RECOMMEND_OUTPUT_RE.search(text or ""))

    @staticmethod
    def _is_recommend_intent(text: str) -> bool:
        """判断用户输入是否为推荐/调整推荐意图（规则兜底，弥补 LLM 识别不稳定）。

        排除规则：
          - 加购/下单意图（命中 _ADD_TO_CART_INTENT_RE）→ 不触发，避免抢 add_to_cart
          - 纯闲聊（命中 _CHAT_INTENT_RE）→ 不触发
        """
        if not text:
            return False
        if _ADD_TO_CART_INTENT_RE.search(text):
            return False
        if _CHAT_INTENT_RE.match(text.strip()):
            return False
        return bool(_RECOMMEND_INTENT_RE.search(text))

    def _fallback_recommend(
        self, user_input: str, history: list, membership_level: str = ""
    ) -> tuple[str, list]:
        """规则兜底：LLM 未调用推荐工具时，强制调用 recommend_dishes 并确定性渲染。

        参数从用户输入 + 历史中轻量提取（不引入额外 LLM 调用）：
          - people_count:   输入中的人数（"6个人" → 6）
          - taste:          口味词（特辣/中辣/微辣/酸辣/香辣/不辣/清淡）
          - customer_type:  人群词（孕妇/儿童/老人/情侣/一人食）
          - exclude_dishes: 历史 AIMessage 中已推荐的菜名（换菜/追加场景排除）
          - membership_level: 透传系统注入的会员等级
        输出经 _clean_tool_output 清理后直接返回，杜绝复述历史/编造菜品。
        """
        text = user_input or ""

        # 人数（支持中文数字："6个人" / "六个人" / "两个人"）
        m = re.search(r"(\d+|[一二两三四五六七八九十百]+)\s*个?人", text)
        if m:
            num_str = m.group(1)
            people_count = int(num_str) if num_str.isdigit() else _chinese_to_digit(num_str)
        else:
            # 未显式提到人数：传 0，由 recommend_dishes 按默认配额处理
            people_count = 0

        # 口味
        taste = ""
        for t in ("特辣", "中辣", "微辣", "酸辣", "香辣"):
            if t in text:
                taste = t
                break
        if not taste and ("不辣" in text or "清淡" in text):
            taste = "不辣"

        # 人群
        customer_type = ""
        for c in ("孕妇", "儿童", "小孩", "老人", "情侣", "一人食"):
            if c in text:
                customer_type = "儿童" if c == "小孩" else c
                break

        # 健康标签（高蛋白/低脂/低糖/素食/无麸质）
        health_tags = ""
        for t in ("高蛋白", "低脂", "低糖", "素食", "无麸质"):
            if t in text:
                health_tags = t
                break

        # 过敏原（海鲜/花生/鸡蛋/牛奶/大豆/大蒜）
        allergen_avoid = ""
        for a in ("海鲜", "花生", "鸡蛋", "牛奶", "大豆", "大蒜"):
            if f"{a}过敏" in text or f"对{a}" in text:
                allergen_avoid = a
                break

        # 历史已推荐菜名 → exclude_dishes（换菜/追加场景排除，避免重复推荐）
        exclude: set[str] = set()
        for msg in history:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                exclude.update(self._extract_dish_entries(content).keys())

        kwargs: dict = {
            "people_count": people_count,
            "taste": taste,
            "customer_type": customer_type,
        }
        if health_tags:
            kwargs["health_tags"] = health_tags
        if allergen_avoid:
            kwargs["allergen_avoid"] = allergen_avoid
        if membership_level:
            kwargs["membership_level"] = membership_level
        if exclude:
            kwargs["exclude_dishes"] = ",".join(sorted(exclude))

        from tools import recommend_dishes
        logger.warning("推荐意图规则兜底: user_input=%s kwargs=%s", user_input[:40], kwargs)
        tool_output = recommend_dishes.invoke(kwargs)
        response = self._clean_tool_output(tool_output)
        return response, [HumanMessage(content=user_input), AIMessage(content=response)]

    @classmethod
    def _extract_dish_entries(cls, text: str) -> dict[str, str]:
        """从推荐文本中提取 {菜名: 价格} 映射"""
        return {name: price for name, price in cls._DISH_ENTRY_RE.findall(text)}

    @classmethod
    def _clean_tool_output(cls, tool_output: str) -> str:
        """清理工具原始输出，去除 [推荐上下文] 部分，使其适合直接展示给用户"""
        idx = tool_output.find("\n[推荐上下文]")
        if idx != -1:
            return tool_output[:idx].rstrip()
        return tool_output

    @staticmethod
    def _find_recommend_output(tool_results: list[str]) -> str:
        """从工具结果中定位 recommend_dishes 的原始输出（含菜品列表与合计）"""
        for r in tool_results:
            if "为您推荐以下菜品" in r and "合计" in r:
                return r
        return ""

    @classmethod
    def _render_recommendation(cls, tool_output: str, polished: RecommendPolished) -> str:
        """确定性渲染推荐结果：开场白 + 工具菜品列表（含合计） + 推荐理由 + 收尾语。

        LLM 只提供语气词文本（opening/reason/closing），菜品/价格/分类/合计全部来自
        工具输出，从结构上杜绝菜名编造、遗漏、价格篡改三类幻觉。
        """
        body = cls._clean_tool_output(tool_output)
        parts = [polished.opening, body]
        if polished.reason:
            # "推荐理由：" 前缀会被前端识别为独立理由区域
            parts.append(f"推荐理由：{polished.reason}")
        parts.append(polished.closing)
        return "\n\n".join(parts)

    @classmethod
    def _validate_reason(cls, reason: str, real_names: set[str]) -> bool:
        """校验推荐理由（reason）白名单：
          1. 为空 → 通过（可无理由）
          2. 严禁出现价格（￥）或"合计"
          3. 提及的菜名必须来自推荐列表（出现未推荐菜单菜名 → 判为幻觉）
        """
        if not reason or not reason.strip():
            return True
        if "￥" in reason or "合计" in reason:
            logger.warning("推荐理由含价格/合计: %s", reason[:30])
            return False
        try:
            from tools import get_merged_dishes
            all_menu_names = {d.name for d in get_merged_dishes()}
        except Exception:
            # 菜单数据不可用时退化为宽松校验（仅禁价格/合计）
            return True
        if cls._mentions_unknown_dish(reason, real_names, all_menu_names):
            logger.warning("推荐理由提及未推荐菜品: %s", reason[:40])
            return False
        return True

    @staticmethod
    def _mentions_unknown_dish(text: str, allowed_names: set[str], all_names: set[str]) -> bool:
        """检测文本中是否出现「允许集合之外」的菜单菜名（推荐理由白名单校验）。

        放行规则：与已推荐菜名存在包含关系的菜名视为简称/变体（如"香茅草烤鱼"
        是"傣味香茅草烤鱼"的简称），不判定为幻觉。

        Returns:
            True 表示文本出现了未被推荐、且与推荐菜无关的真实菜单菜名（视为幻觉）
        """
        if not text:
            return False
        for name in all_names:
            if name in allowed_names:
                continue
            # 简称/变体放行：与某个已推荐菜名存在包含关系
            if any(name in a or a in name for a in allowed_names):
                continue
            if name in text:
                return True
        return False

    @staticmethod
    def _mentions_dish(text: str, dish_names: set[str]) -> bool:
        """检测文本是否出现任何菜名。

        用于双保险校验：LLM 生成的语气词（开场白/收尾语）中严禁复述菜名。
        """
        if not dish_names:
            return False
        return any(name and name in text for name in dish_names)

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        """容错解析 LLM 返回的 JSON 对象（支持 markdown ```json 包裹与首尾噪音）。

        Returns:
            dict 或 None（解析失败）
        """
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    def _json_rewrite(self, messages: list, json_instruction: str, model: type[BaseModel]):
        """调用 LLM 生成结构化（JSON Mode）输出并解析为 Pydantic 模型。

        防幻觉设计：
          - 强制 response_format={"type":"json_object"}（DeepSeek 官方保证输出合法 JSON）。
          - schema 内嵌于 json_instruction，由模型严格遵循。
          - 解析/校验失败时返回 None，调用方回退到确定性渲染（工具原始输出）。
        """
        try:
            final_msg = self.llm_json.invoke(
                messages + [SystemMessage(content=json_instruction)]
            )
            data = self._parse_json_object(final_msg.content or "")
            if data is None:
                return None
            return model(**data)
        except Exception as e:
            logger.warning("LLM 结构化输出解析失败（回退确定性渲染）: %s", e)
            return None

    @staticmethod
    def _is_complete_response(text: str) -> bool:
        """判断工具结果是否为完整回复，无需再调用 LLM。

        recommend_dishes/query_dish/list_menu 等工具返回的格式完整的内容，
        直接展示给用户即可，LLM 复述反而增加延迟和幻觉风险。
        """
        if not text or len(text) < 50:
            return False
        indicators = ["为您推荐", "合计", "￥", "菜品信息", "菜单列表",
                      "未找到", "搭配方案", "过敏原", "互斥规则",
                      "已加入购物车", "加购失败", "未识别到要加购"]
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

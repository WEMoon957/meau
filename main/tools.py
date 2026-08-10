"""工具函数模块 - 使用 LangChain @tool 装饰器定义工具

菜品查询、智能推荐、菜品知识库查询，供 LangChain Agent 调用。
"""

import contextvars
import copy
import logging
from langchain_core.tools import tool

from menu_data import Dish, get_all_dishes

logger = logging.getLogger("tools")


# ======================== 会话上下文（供 add_to_cart 工具拿凭证+历史） ========================
# 方案 A：统一入口 + LLM 路由后，add_to_cart 工具由 LLM 在 agent.chat 内触发，
# 但工具函数无法直接拿到 session_id/session_token/history。用 contextvars 在
# SessionManager.chat 设置当前请求的会话凭证与历史，工具内部读取。
# contextvars 在 asyncio.to_thread 中自动传播，线程安全。
_session_ctx: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "_session_ctx", default=None
)
# 会话历史（list[HumanMessage|AIMessage]）：供 add_to_cart 在"确认下单"场景
# 从最近推荐中提取菜名。由 SessionManager.chat 在加载历史后设置。
_history_ctx: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "_history_ctx", default=None
)


def set_session_context(session_id: str, session_token: str) -> contextvars.Token:
    """设置当前请求的会话凭证（api_server.ai_chat 调用）。

    Returns:
        contextvars.Token，用于 reset 恢复上下文（请求结束清理）。
    """
    return _session_ctx.set((session_id, session_token))


def reset_session_context(token: contextvars.Token) -> None:
    """恢复会话上下文（请求结束清理，避免上下文泄漏到下一个请求）。"""
    _session_ctx.reset(token)


def set_history_context(history: list) -> contextvars.Token:
    """设置当前请求的会话历史（SessionManager.chat 加载历史后调用）。

    供 add_to_cart 工具在"确认下单"场景从最近推荐中提取菜名。
    """
    return _history_ctx.set(history)


def reset_history_context(token: contextvars.Token) -> None:
    """恢复历史上下文（请求结束清理）。"""
    _history_ctx.reset(token)


def _extract_dish_names_from_history(history: list) -> list[str]:
    """从会话历史中提取最近一次推荐的菜名。

    推荐结果的 AIMessage 格式为"菜名  ￥价格"，用正则提取。
    从后往前找第一条包含"￥"的 AIMessage。
    """
    import re
    for msg in reversed(history):
        content = getattr(msg, "content", "") or ""
        if "￥" not in content:
            continue
        # 匹配"菜名  ￥价格"格式（菜名前可能有序号/缩进）
        names = re.findall(r"([\u4e00-\u9fa5\w·]+)\s+￥\d+", content)
        if names:
            # 过滤掉明显不是菜名的（如"合计"）
            return [n.strip() for n in names if n.strip() and n != "合计"]
    return []


# ======================== 双源数据合并（向量库 + MySQL） ========================
# 向量库（ChromaDB）的 dish_profile 是辣度/过敏原/人群的权威数据源，
# MySQL 的 dishes 表是价格/分类/毛利率的结构化数据源。
# 推荐前必须合并双源，用向量库字段补全 MySQL 的空字段，避免过滤失效。

# 向量库过敏原类别到 MySQL 存储值的映射
_KB_ALLERGEN_CATEGORIES = ["香菜", "葱", "蒜", "花生", "海鲜", "乳制品", "鸡蛋", "大豆", "麸质", "坚果"]
# 向量库"乳制品"对应 MySQL"牛奶"
_KB_ALLERGEN_ALIASES = {"乳制品": "牛奶", "蒜": "大蒜"}

# 合并后的双源缓存（按菜名索引），避免每次推荐都重复合并
_merged_dishes_cache: dict[str, Dish] | None = None
# 向量库已验证过敏原的菜名集合（allergen_info 非空的菜品）
# 这些菜品的过敏原信息以向量库为准，不再走菜名关键词兜底（避免误杀）
_kb_allergen_verified: set[str] = set()


def _parse_kb_allergen_info(allergen_info: str) -> list[str]:
    """解析向量库的 allergen_info 文本，返回含过敏原的 MySQL 标准名列表

    向量库格式："不含香菜，不含葱，不含蒜，不含花生，不含海鲜，不含乳制品"
    或："含花生，不含海鲜"

    Returns:
        含过敏原的菜品返回 ["花生"]，不含的返回 []
    """
    if not allergen_info:
        return []
    contains = []
    for kb_cat in _KB_ALLERGEN_CATEGORIES:
        # 检查"含X"且非"不含X"
        contain_marker = f"含{kb_cat}"
        exclude_marker = f"不含{kb_cat}"
        if contain_marker in allergen_info and exclude_marker not in allergen_info:
            # 转换为 MySQL 标准名
            mysql_name = _KB_ALLERGEN_ALIASES.get(kb_cat, kb_cat)
            contains.append(mysql_name)
    return contains


def _parse_kb_spice_level(spice_level: str) -> str:
    """归一化向量库的辣度到标准枚举

    向量库可能存"微辣(芥末)"等扩展值，需归一到"微辣"
    """
    if not spice_level:
        return ""
    for std in ["不辣", "微辣", "中辣", "特辣"]:
        if std in spice_level:
            return std
    return ""


def _parse_kb_suitable_crowd(suitable_crowd: str) -> list[str]:
    """解析向量库的 suitable_crowd 到 MySQL suitable_for 格式

    向量库格式："老人、小孩" → ["老人", "小孩"]
    注意：向量库用"小孩"，MySQL 用"儿童"，需统一
    """
    if not suitable_crowd:
        return []
    # 按顿号/逗号分隔
    import re
    parts = re.split(r"[、,，]", suitable_crowd)
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # "小孩" → "儿童"
        if p == "小孩":
            p = "儿童"
        result.append(p)
    return result


def _parse_kb_dietary_tags(calorie: str, property_val: str) -> list[str]:
    """解析向量库的 calorie 和 property 字段，映射为系统饮食标签

    向量库字段：
      - calorie: "低热量" / "极低热量" / "中热量" 等
      - property: "素菜" / "荤菜" 等

    映射规则（仅当 MySQL dietary_tags 为空时补全）：
      - "低热量" / "极低热量" → "低脂"
      - "素菜" / 含"素" → "素食"
      - "荤菜" / 含"荤" → "高蛋白"

    注意：系统合法标签为 素食/低脂/低糖/高蛋白/无麸质，不生成非法标签。
    """
    tags = []
    if calorie and ("低热量" in calorie or "极低热量" in calorie):
        tags.append("低脂")
    if property_val:
        if "素" in property_val:
            tags.append("素食")
        elif "荤" in property_val:
            tags.append("高蛋白")
    return tags


def _build_kb_description(kb_meta: dict) -> str:
    """从向量库元数据构建菜品描述文本

    组合 property（荤素）、calorie（热量）、salt_level（咸度）、pairing（搭配建议）
    生成简洁的描述文本，仅在 MySQL description 为空时使用。
    """
    parts = []
    prop = kb_meta.get("property", "")
    calorie = kb_meta.get("calorie", "")
    salt_level = kb_meta.get("salt_level", "")
    pairing = kb_meta.get("pairing", "")

    if prop:
        parts.append(prop)
    if calorie:
        parts.append(calorie)
    if salt_level:
        parts.append(salt_level)
    if pairing:
        parts.append(f"搭配建议：{pairing}")

    return "；".join(parts) if parts else ""


def _load_kb_profiles() -> dict:
    """从向量库加载所有 dish_profile，按菜名索引

    Returns:
        {菜名: {spice_level, allergen_info, suitable_crowd, ...}}
        失败时返回空字典（静默降级，不阻塞推荐）
    """
    try:
        import kb_query
        profiles = kb_query._get_kb().get_all_by_type("dish_profile")
        result = {}
        for p in profiles:
            meta = p.get("metadata", {})
            name = meta.get("dish_name", "")
            if name:
                result[name] = meta
        return result
    except Exception as e:
        # 静默降级：向量库不可用时回退到纯 MySQL 数据
        print(f"⚠️ 向量库加载失败，回退到纯MySQL数据: {e}")
        return {}


def _merge_dish_with_kb(dish: Dish, kb_meta: dict) -> Dish:
    """用向量库字段补全 MySQL Dish 对象的空字段

    合并规则（向量库为权威源，仅在 MySQL 字段为空时补全）：
    - spicy_level:  MySQL 空 → 用向量库 spice_level
    - allergens:    MySQL 空 → 用向量库 allergen_info 解析结果
    - suitable_for: MySQL 空 → 用向量库 suitable_crowd 解析结果
    - dietary_tags: MySQL 空 → 用向量库 calorie + property 推导
    - description:  MySQL 空 → 用向量库 property/calorie/salt/pairing 拼接
    - category:     MySQL 空 → 用向量库 category 补全

    返回新的 Dish 对象（深拷贝，不修改原对象）
    """
    merged = copy.deepcopy(dish)

    # 辣度补全
    if not merged.spicy_level or merged.spicy_level == "":
        kb_spicy = _parse_kb_spice_level(kb_meta.get("spice_level", ""))
        if kb_spicy:
            merged.spicy_level = kb_spicy

    # 过敏原补全（MySQL allergens 为空列表时用向量库补全）
    if not merged.allergens:
        kb_allergens = _parse_kb_allergen_info(kb_meta.get("allergen_info", ""))
        if kb_allergens:
            merged.allergens = kb_allergens

    # 人群补全
    if not merged.suitable_for:
        kb_crowd = _parse_kb_suitable_crowd(kb_meta.get("suitable_crowd", ""))
        if kb_crowd:
            merged.suitable_for = kb_crowd

    # 饮食标签补全：MySQL dietary_tags 可能包含迁移时产生的无效标签名
    # （如 migrate_menu.py 写入的 "低热量"/"素菜"/"荤菜"），需做归一化处理
    _VALID_TAGS = {"素食", "低脂", "低糖", "高蛋白", "无麸质"}
    existing_valid = [t for t in merged.dietary_tags if t in _VALID_TAGS]
    kb_tags = _parse_kb_dietary_tags(
        kb_meta.get("calorie", ""),
        kb_meta.get("property", ""),
    )
    # 策略：MySQL 有效标签 + KB 推导标签 取并集（去重）
    merged_tags = list(set(existing_valid + kb_tags))
    if merged_tags:
        merged.dietary_tags = merged_tags

    # 描述补全（从向量库元数据拼接）
    if not merged.description or merged.description == "":
        kb_desc = _build_kb_description(kb_meta)
        if kb_desc:
            merged.description = kb_desc

    # 分类补全
    if not merged.category or merged.category == "":
        kb_cat = kb_meta.get("category", "")
        if kb_cat:
            merged.category = kb_cat

    return merged


def get_merged_dishes() -> list[Dish]:
    """获取双源合并后的菜品列表（向量库 + MySQL）

    流程：
    1. 从 MySQL 加载所有菜品（结构化数据：价格、分类、毛利率）
    2. 从向量库加载所有 dish_profile（权威属性：辣度、过敏原、人群）
    3. 按菜名匹配，用向量库字段补全 MySQL 的空字段
    4. 返回合并后的 Dish 列表

    缓存：合并结果缓存在 _merged_dishes_cache，避免重复 I/O
    调用 invalidate_merged_cache() 可清除缓存

    Returns:
        合并后的 Dish 列表，向量库不可用时返回纯 MySQL 数据
    """
    global _merged_dishes_cache, _kb_allergen_verified
    if _merged_dishes_cache is not None:
        return list(_merged_dishes_cache.values())

    mysql_dishes = get_all_dishes()
    kb_profiles = _load_kb_profiles()

    # 记录向量库已验证过敏原的菜名（allergen_info 非空即视为已验证）
    verified = set()
    merged_list = []
    merged_dict = {}
    for dish in mysql_dishes:
        kb_meta = kb_profiles.get(dish.name, {})
        if kb_meta.get("allergen_info"):
            verified.add(dish.name)
        merged = _merge_dish_with_kb(dish, kb_meta)
        merged_list.append(merged)
        merged_dict[dish.name] = merged

    _merged_dishes_cache = merged_dict
    _kb_allergen_verified = verified
    return merged_list


def invalidate_merged_cache():
    """清除双源合并缓存（数据更新后调用）"""
    global _merged_dishes_cache, _kb_allergen_verified
    _merged_dishes_cache = None
    _kb_allergen_verified = set()



# ======================== 推荐算法辅助 ========================

def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _filter_by_taste(candidates: list[Dish], taste: str) -> list[Dish]:
    if not taste or taste == "不限":
        return candidates
    if taste == "不辣":
        return [d for d in candidates if d.spicy_level == "不辣"]
    if taste in ("微辣", "中辣", "特辣"):
        return [d for d in candidates if d.spicy_level == taste]
    if "辣" in taste:
        return [d for d in candidates if d.spicy_level != "不辣"]
    return candidates


# 过敏原菜名关键词兜底映射：数据标注不完整时的安全网
# 当 allergens 字段未标注但菜名含明确关键词时，按过敏原类别兜底过滤
_ALLERGEN_NAME_KEYWORDS = {
    "海鲜": ("鱼", "虾", "蟹", "贝", "螺", "鱿鱼", "墨鱼", "章鱼", "扇贝", "蛤", "蚝", "鲍"),
    "花生": ("花生",),
    "鸡蛋": ("鸡蛋", "蛋"),
    "牛奶": ("牛奶", "芝士", "黄油", "奶油", "双皮奶"),
    "大豆": ("大豆", "豆腐", "豆干", "豆浆"),
    "大蒜": ("大蒜", "蒜蓉"),
}


def _filter_by_allergens(candidates: list[Dish], avoid_list: list[str]) -> list[Dish]:
    """按过敏原过滤候选菜品

    过滤优先级：
    1. allergens 字段过滤（双源合并后的权威值）
    2. 菜名关键词兜底：仅当双源都无过敏原数据时启用（菜名不在 _kb_allergen_verified 中）

    注意：向量库已明确标注过敏原的菜品（含"不含X"），不再走菜名兜底，
    避免误杀"傣味香茅草烤鱼（不含海鲜）"这类向量库已澄清的菜品。
    """
    if not avoid_list:
        return candidates
    result = []
    for d in candidates:
        # 1. 按 allergens 字段过滤（主过滤，双源合并后的权威值）
        if any(allergen in d.allergens for allergen in avoid_list):
            continue
        # 2. 菜名兜底：仅当双源都无过敏原数据时启用
        #    向量库已验证的菜品（allergen_info 非空）跳过兜底，避免误杀
        if d.name not in _kb_allergen_verified:
            blocked = False
            for allergen in avoid_list:
                keywords = _ALLERGEN_NAME_KEYWORDS.get(allergen)
                if keywords and any(kw in d.name for kw in keywords):
                    blocked = True
                    break
            if blocked:
                continue
        result.append(d)
    return result


# 人群标签白名单：只有这些值才用于 suitable_for 硬筛选
# 场景词（聚餐/情侣/一人食/家庭）不参与硬筛选，避免把候选过滤光
_DEMOGRAPHIC_TAGS = {"青年", "小孩", "儿童", "老人", "孕妇"}


def _filter_by_customer_type(candidates: list[Dish], customer_type: str) -> list[Dish]:
    """按人群类型筛选候选菜品

    只对真实人群标签（青年/小孩/老人/儿童/孕妇）做硬筛选；
    场景词（聚餐/情侣/一人食/家庭聚餐等）不筛选，避免把候选过滤光。

    特例：菜名含"儿童"的菜品，其 suitable_for 常为空，但当 customer_type
    为"小孩/儿童"时应保留（由后续 _filter_by_scene 处理）。
    """
    if not customer_type:
        return candidates
    # 归一化：小孩 → 儿童（数据中 suitable_for 统一存"儿童"，见 _parse_kb_suitable_crowd）
    tag = customer_type.strip()
    if tag == "小孩":
        tag = "儿童"
    if tag not in _DEMOGRAPHIC_TAGS:
        # 场景词，不做硬筛选
        return candidates
    # 人群标签硬筛选，但菜名含"儿童"的菜品豁免（交给场景过滤处理）
    result = []
    for d in candidates:
        if "儿童" in d.name:
            result.append(d)
            continue
        if tag in d.suitable_for:
            result.append(d)
    return result


# 菜名含这些关键词的菜品，仅在 customer_type 匹配时才纳入推荐
# 避免给成年人聚餐推荐儿童餐等场景不匹配的情况
_SPECIAL_DISH_KEYWORDS = {
    "儿童": {"儿童", "小孩"},
}

# 多人份菜品关键词 → 适用最低人数
# 含"四人"的拼盘至少3人才能推荐，含"双人"的至少2人
_PARTY_SIZE_KEYWORDS = {
    "四人": 3,
    "双人": 2,
}


def _filter_by_scene(candidates: list[Dish], customer_type: str) -> list[Dish]:
    """按场景过滤特殊菜品

    菜名含"儿童"的菜品，仅当 customer_type 含"儿童/小孩"时才保留。
    成年人聚餐/情侣/一人食等场景不推荐儿童餐。
    """
    if not candidates:
        return candidates
    # 归一化
    ct = (customer_type or "").strip()
    if ct == "儿童":
        ct = "小孩"

    filtered = []
    for d in candidates:
        should_skip = False
        for keyword, allowed_tags in _SPECIAL_DISH_KEYWORDS.items():
            if keyword in d.name:
                # 菜名含特殊关键词，检查 customer_type 是否匹配
                if ct not in allowed_tags:
                    should_skip = True
                    break
        if not should_skip:
            filtered.append(d)
    return filtered


def _filter_by_party_size(candidates: list[Dish], people_count: int) -> list[Dish]:
    """按人数过滤多人份拼盘菜品

    菜名含"四人/双人"等关键词的菜品，人数不足时排除：
      - 含"四人"：至少 3 人才能推荐
      - 含"双人"：至少 2 人才能推荐

    避免给单人推荐四人拼盘这类不合理情况。
    """
    if not candidates or people_count <= 0:
        return candidates

    filtered = []
    for d in candidates:
        should_skip = False
        for keyword, min_people in _PARTY_SIZE_KEYWORDS.items():
            if keyword in d.name:
                if people_count < min_people:
                    should_skip = True
                    break
        if not should_skip:
            filtered.append(d)
    return filtered


def _dish_score(d: Dish, weather_set: set[int], season_set: set[int]) -> float:
    """菜品推荐评分（满分 25，毛利率占 20%）

    评分构成：
      - 招牌菜:    10 分 (40%)
      - 天气匹配:  5 分 (20%)
      - 季节匹配:  3 分 (12%)
      - 毛利率:    5 分 (20%)  毛利率 0.60→0, 1.00→5 线性映射
      - 价格合理:  2 分 (8%)   价格落在 20~80 元区间加分
    """
    score = 0.0
    if d.is_signature:
        score += 10
    if id(d) in weather_set:
        score += 5
    if id(d) in season_set:
        score += 3
    # 毛利率权重 20%：0.60 以下记 0 分，1.00 记 5 分
    gm = d.gross_margin or 0.0
    score += max(0.0, min(5.0, (gm - 0.60) / 0.40 * 5.0))
    # 价格合理性：20~80 元区间加分
    if 20 <= d.price <= 80:
        score += 2
    return score


def _total_quota(people_count: int) -> int:
    """根据人数返回推荐总数

    一人食: 3 道
    2人食: 5 道
    3-4人: 8 道
    5-8人: 12 道
    8人以上: 16 道
    """
    if people_count <= 0:
        return 5
    if people_count <= 1:
        return 3
    if people_count <= 2:
        return 5
    if people_count <= 4:
        return 8
    if people_count <= 8:
        return 12
    return 16


# 同分类最多选几道，避免推荐全是同一分类
_MAX_PER_CATEGORY = 2


# ======================== LangChain 工具定义 ========================

@tool
def query_dish(dish_name: str) -> str:
    """查询菜品详细信息，包括价格、辣度、适合人群、过敏原、饮食标签等。当顾客询问某道菜的具体信息时使用。

    使用参数化查询数据库，精确匹配优先，模糊匹配兜底。

    Args:
        dish_name: 菜品名称，如：包烧见手青、松茸刺身、牛肝菌焖饭
    """
    from text_to_sql import query_dish_by_name
    return query_dish_by_name(dish_name)


@tool
def list_menu(category: str = "") -> str:
    """列出菜单菜品，可按分类筛选。当顾客想看菜单或浏览某类菜品时使用。

    使用参数化查询数据库，按分类白名单筛选。

    Args:
        category: 菜品分类：菌彩特色/进店必点/经典推荐/菌汤锅底/山珍菌宴/云岭特色/山茅野菜/普洱黄牛肉/涮品/甜饮品，为空则列出全部
    """
    from text_to_sql import list_menu_dishes
    return list_menu_dishes(category)


@tool
def recommend_dishes(
    people_count: int = 0,
    taste: str = "",
    customer_type: str = "",
    health_tags: str = "",
    weather: str = "",
    season: str = "",
    allergen_avoid: str = "",
    include_drinks: bool = True,
    include_staple: bool = True,
    include_soup: bool = True,
    exclude_dishes: str = "",
    membership_level: str = "",
) -> str:
    """根据顾客需求智能推荐菜品。采用全新的多样化推荐算法，确保每次推荐都有差异性。
    
    推荐策略：
    1. 强制包含锅底（火锅店业务要求）
    2. 多维度评分 + 随机化选择
    3. 分类均衡搭配
    4. 动态权重调整
    5. 会员等级加权（会员等级越高，越倾向推荐高评分招牌菜和特色菜）
    
    Args:
        people_count: 用餐人数，0表示不限制
        taste: 口味偏好：不辣/微辣/中辣/特辣/酸辣/香辣/不限
        customer_type: 人群类型：儿童/老人/聚餐/情侣/一人食
        health_tags: 健康标签，多个用逗号分隔：低脂/低糖/高蛋白/素食/无麸质
        weather: 当前天气：热天/冷天/雨天
        season: 当前季节：春/夏/秋/冬
        allergen_avoid: 需要避开的过敏原，多个用逗号分隔：花生/海鲜/鸡蛋/牛奶/大豆
        include_drinks: 是否包含饮品（甜饮品分类），默认True
        include_staple: 是否包含主食类菜品，默认True
        include_soup: 是否包含汤锅类（菌汤锅底分类），默认True
        exclude_dishes: 需要排除的菜品名称（已推荐过的），多个用逗号分隔
        membership_level: 会员等级：普通会员/银卡会员/金卡会员/钻石会员
    """
    import random
    random.seed()  # 使用系统时间作为随机种子

    # 双源合并：MySQL 结构化数据 + 向量库权威属性（辣度/过敏原/人群）
    # 向量库是过敏原的权威源，避免 MySQL 字段空导致过滤失效
    all_dishes = get_merged_dishes()
    candidates = list(all_dishes)
    avoid_list = _parse_csv(allergen_avoid)
    exclude_set = {x.strip() for x in _parse_csv(exclude_dishes) if x.strip()}
    
    # ========== 基础筛选 ==========
    # 推荐总数（提前计算，供软筛选候选不足兜底使用）
    total_quota = _total_quota(people_count)

    # 1. 口味筛选（带渐进降级：特辣→中辣→微辣→不限）
    original_taste = taste
    candidates = _filter_by_taste(candidates, taste)
    if not candidates and taste in ("特辣", "中辣", "微辣"):
        # 特辣无结果 → 降级到中辣及以上
        relaxed = "中辣" if taste == "特辣" else "微辣"
        candidates = [d for d in all_dishes if d.spicy_level in (taste, relaxed)]
        if candidates:
            taste = f"{taste}/{relaxed}"  # 标记实际使用的口味
    if not candidates and taste in ("特辣/中辣", "中辣/微辣"):
        # 仍无结果 → 降级到所有辣味
        candidates = [d for d in all_dishes if d.spicy_level != "不辣"]
        if candidates:
            taste = "辣"
    if not candidates:
        # 完全无辣味菜品 → 取消口味限制
        candidates = list(all_dishes)
        taste = ""
    
    # 2. 人群筛选（带兜底：候选不足时放宽人群硬筛选；场景过滤仍防止儿童餐泄漏）
    candidates = _filter_by_customer_type(candidates, customer_type)
    if len(candidates) < total_quota:
        # 数据稀疏（如"孕妇"适合人群标注为 0 道）时放宽人群限制，避免推荐只剩锅底
        candidates = _filter_by_customer_type(list(all_dishes), "")

    # 3. 场景过滤（儿童餐等）
    candidates = _filter_by_scene(candidates, customer_type)

    # 3.5. 按人数过滤多人份拼盘（四人拼盘至少3人，双人拼盘至少2人）
    candidates = _filter_by_party_size(candidates, people_count)

    # 4. 健康标签筛选（多标签 OR 语义 + 兜底：候选不足时忽略健康限制）
    if health_tags:
        pre_health = candidates
        tags = _parse_csv(health_tags)
        candidates = [d for d in candidates
                      if any(tag in d.dietary_tags for tag in tags)]
        if len(candidates) < total_quota:
            # 数据稀疏（如"高蛋白"标注为 0 道）时忽略健康限制，避免推荐只剩锅底
            candidates = pre_health
    
    # 5. 过敏原排除
    candidates = _filter_by_allergens(candidates, avoid_list)
    
    # 6. 品类开关过滤
    if not include_drinks:
        candidates = [d for d in candidates if d.category != "甜饮品"]
    
    candidates = [d for d in candidates
                  if d.category != "菌汤锅底" or "锅" in d.name]
    
    if not include_staple:
        staple_keywords = ("饵丝", "饵块", "焖饭", "糯米粉", "米线", "面")
        candidates = [d for d in candidates
                      if not any(kw in d.name for kw in staple_keywords)]
    
    # 7. 排除已推荐菜品（追加场景）
    if exclude_set:
        candidates = [d for d in candidates
                      if d.name not in exclude_set
                      or (d.category == "菌汤锅底" and "锅" in d.name)]
    
    # ========== 推荐逻辑 ==========

    recommended = []
    used_names = set()
    cat_count = {}
    
    # ========== 锅底强制推荐 ==========
    
    pot_pool = _filter_by_allergens(
        [d for d in all_dishes if d.category == "菌汤锅底" and "锅" in d.name],
        avoid_list,
    )
    
    if pot_pool:
        # 选择锅底：优先未排除的，否则随机选择
        available_pots = [d for d in pot_pool if d.name not in exclude_set]
        if available_pots:
            selected_pot = random.choice(available_pots)
        else:
            selected_pot = random.choice(pot_pool)
        
        recommended.append(selected_pot)
        used_names.add(selected_pot.name)
        cat_count[selected_pot.category] = 1
    
    # ========== 多样化推荐算法 ==========
    
    # 将候选池按分类分组
    category_pools = {}
    for dish in candidates:
        if dish.name not in used_names:
            if dish.category not in category_pools:
                category_pools[dish.category] = []
            category_pools[dish.category].append(dish)
    
    # 为每个分类计算动态权重
    category_weights = _calculate_category_weights(category_pools, customer_type, weather, season, membership_level)
    
    # 多轮选择，确保分类均衡
    # 规则引擎：跟踪被冲突规则跳过的菜品名称
    skipped_by_rules = []
    for round_num in range(3):  # 最多3轮选择
        if len(recommended) >= total_quota:
            break

        # 每轮重新计算剩余配额（P0修复：原代码remaining_slots只算一次，导致后续轮次可能超额）
        remaining_slots = total_quota - len(recommended)

        # 根据轮次调整选择策略
        if round_num == 0:
            # 第一轮：优先高权重分类，每个分类最多2个
            _select_by_category_weights(category_pools, category_weights, recommended,
                                     used_names, cat_count, max_per_category=2,
                                     remaining_slots=remaining_slots, total_quota=total_quota,
                                     skipped_by_rules=skipped_by_rules)
        elif round_num == 1:
            # 第二轮：放宽分类限制，但保持多样性
            _select_by_category_weights(category_pools, category_weights, recommended,
                                     used_names, cat_count, max_per_category=3,
                                     remaining_slots=remaining_slots, total_quota=total_quota,
                                     skipped_by_rules=skipped_by_rules)
        else:
            # 第三轮：从剩余菜品中随机选择
            _select_from_remaining(category_pools, recommended, used_names,
                                 remaining_slots, total_quota,
                                 skipped_by_rules=skipped_by_rules)
    
    # ========== 饮品保障 ==========
    # 如果 include_drinks=True 但推荐结果中没有饮品，尝试替换最后一道菜为饮品
    if include_drinks and len(recommended) > 1:
        has_drink = any(
            d.category == "甜饮品"
            or any(kw in d.name for kw in ("茶", "酒", "莓"))
            for d in recommended
        )
        if not has_drink:
            # 从候选池找饮品（不限分类，饮料类关键词或甜饮品分类）
            drink_pool = [d for d in candidates
                          if d.name not in used_names
                          and (d.category == "甜饮品"
                               or any(kw in d.name for kw in ("茶", "酒", "莓")))]
            if drink_pool:
                # 替换最后一个非锅底菜品
                for i in range(len(recommended) - 1, -1, -1):
                    if not (recommended[i].category == "菌汤锅底" and "锅" in recommended[i].name):
                        old_dish = recommended[i]
                        new_drink = random.choice(drink_pool)
                        recommended[i] = new_drink
                        used_names.discard(old_dish.name)
                        used_names.add(new_drink.name)
                        cat_count[old_dish.category] = max(0, cat_count.get(old_dish.category, 1) - 1)
                        cat_count[new_drink.category] = cat_count.get(new_drink.category, 0) + 1
                        break
    
    # ========== 结果格式化 ==========
    
    # 按分类分组展示
    cat_order = []
    for d in recommended:
        if d.category not in cat_order:
            cat_order.append(d.category)
    
    lines = ["为您推荐以下菜品：\n"]
    
    for category in cat_order:
        category_dishes = [d for d in recommended if d.category == category]
        
        if category_dishes:
            lines.append(f"\n--- {category} ---")
            for i, dish in enumerate(category_dishes, 1):
                # 调整序号，避免从1开始重新计数
                global_idx = sum(len([d for d in recommended if d.category == cat]) 
                               for cat in cat_order[:cat_order.index(category)]) + i
                lines.append(f"  {global_idx}. {dish.name}  ￥{dish.price}")
                if hasattr(dish, 'spicy_level') and dish.spicy_level:
                    lines.append(f"     [{dish.spicy_level}]")
    
    # 计算总价
    total_price = sum(d.price for d in recommended)
    lines.append(f"\n合计：￥{total_price}")
    
    # 结构化上下文（供 LLM 二次生成口语化推荐理由）
    ctx = _build_recommend_context(taste, customer_type, weather, season,
                                   allergen_avoid, people_count, membership_level,
                                   recommended, skipped_by_rules)
    lines.append(ctx)
    
    return "\n".join(lines)


# ======================== 菜品知识库工具（基于向量检索） ========================

def _format_kb_results(results: list[dict], title: str) -> str:
    """格式化知识库检索结果为可读文本"""
    if not results:
        return f"未找到与「{title}」相关的知识库内容。"

    lines = [f"为您找到 {len(results)} 条相关内容：\n"]
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        text = r.get("text", "")
        meta = r.get("metadata", {})
        meta_type = meta.get("type", "")

        # 根据类型添加标签
        tag = ""
        if meta_type == "dish_profile":
            dish = meta.get("dish_name", "")
            tag = f"【菜品档案】{dish}"
        elif meta_type == "combo_plan":
            tag = f"【搭配方案】{meta.get('section', '')}"
        elif meta_type == "avoid_combo":
            tag = f"【避雷搭配】{meta.get('section', '')}"
        elif meta_type == "mutual_exclusion":
            tag = f"【互斥规则】{meta.get('section', '')}"
        elif meta_type == "fruit_allergen":
            tag = f"【水果过敏】{meta.get('fruit', '')}"
        elif meta_type == "level_definition":
            tag = f"【等级定义】{meta.get('category', '')}"
        else:
            tag = f"【{meta_type}】"

        lines.append(f"[{i}] {tag}（相关度: {score:.2f}）")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


@tool
def search_dish_knowledge(query: str) -> str:
    """从菜品知识库中语义搜索菜品相关信息。知识库包含83种菜品的完整档案（辣度、咸度、热量、适合人群、过敏原）、搭配方案、互斥规则、水果过敏原等。当顾客询问菜品属性、搭配建议、过敏原、口味冲突等问题时使用。

    与 query_dish 的区别：query_dish 查询数据库中的菜品价格和基本信息；本工具搜索知识库中的菜品属性详情（辣度咸度分级、热量等级、适合人群、过敏原标记等）。

    适用场景：
    - "这道菜辣不辣/咸不咸"
    - "适合老人/小孩吃吗"
    - "有没有香菜/花生/海鲜"
    - "热量高不高"
    - 任何需要菜品属性详情的问题

    Args:
        query: 查询问题，如"适合老人吃不辣的菜""有没有含花生的菜"
    """
    from kb_query import search_dish_profiles
    results = search_dish_profiles(query, top_k=8)
    return _format_kb_results(results, query)


@tool
def get_pairing_plan(query: str) -> str:
    """从知识库中搜索菜品搭配方案。知识库包含4套预设套餐方案（经典地道、酸辣傣味、清淡养生、肉食爱好者），以及避雷搭配提示。当顾客询问聚餐搭配、套餐推荐、几个人怎么点菜等问题时使用。

    适用场景：
    - "3-4人聚餐怎么点"
    - "有什么套餐推荐"
    - "清淡养生的搭配"
    - "什么菜不能一起点"（避雷搭配）

    Args:
        query: 查询问题，如"3-4人聚餐推荐""清淡养生组合""什么不能一起点"
    """
    from kb_query import search_combo_plans, search_avoid_combos

    # 判断是否查询避雷搭配
    if any(kw in query for kw in ["避雷", "不能一起", "不要", "冲突", "不可以"]):
        results = search_avoid_combos(query, top_k=3)
        return _format_kb_results(results, query)

    results = search_combo_plans(query, top_k=3)
    return _format_kb_results(results, query)


@tool
def get_exclusion_rules(query: str) -> str:
    """从知识库中搜索菜品互斥规则（菌子重复、口味冲突）。当顾客同时点了多道菌子、或询问哪些食材不能搭配时使用，避免口味冲突和重复点单。

    适用场景：
    - "这些菌子能一起煮吗"
    - "哪些菌子口味会冲突"
    - "菌子重复了怎么办"
    - "牛肉和腊肉能一起点吗"

    Args:
        query: 查询问题，如"哪些菌子不能一起点""菌子口味冲突"
    """
    from kb_query import search_exclusion_rules
    results = search_exclusion_rules(query, top_k=5)
    return _format_kb_results(results, query)


@tool
def get_fruit_allergen_info(query: str) -> str:
    """从知识库中搜索水果过敏原信息。野生菌火锅与部分水果同食可能引发过敏或不适，本工具提供风险等级和食用建议。当顾客询问吃完菌子能否吃水果、水果过敏等问题时使用。

    适用场景：
    - "吃完菌子能吃芒果吗"
    - "什么水果不能和菌子一起吃"
    - "水果过敏风险"
    - "吃完火锅可以吃水果吗"

    Args:
        query: 查询问题，如"吃完菌子能吃芒果吗""水果过敏"
    """
    from kb_query import search_fruit_allergens
    results = search_fruit_allergens(query, top_k=5)
    return _format_kb_results(results, query)


# ======================== 多样化推荐算法辅助函数 ========================

def _calculate_category_weights(category_pools, customer_type, weather, season, membership_level=""):
    """计算各分类的动态权重（含会员等级加权）"""
    import random
    weights = {}
    
    # 基础权重
    base_weights = {
        "菌汤锅底": 0.0,  # 锅底已单独处理
        "菌彩特色": 1.2,
        "山珍菌宴": 1.0,
        "进店必点": 1.5,
        "经典推荐": 1.0,
        "涮品": 0.8,
        "甜饮品": 1.0,
        "云岭特色": 0.9,
        "山茅野菜": 0.7,
        "普洱黄牛肉": 0.8,
        "鲜切牛肉": 0.85,
        "热炒": 0.9,
        "凉菜": 0.85,
        "菌子": 0.8,
        "锅底": 0.0,  # 锅底（非菌汤锅底）不参与评分
    }
    
    # 会员等级权重加成（乘法因子，叠加到各分类基础权重之上）
    membership_boost = {
        "普通会员": {"进店必点": 1.0, "菌彩特色": 1.0, "经典推荐": 1.0},
        "银卡会员": {"进店必点": 1.15, "菌彩特色": 1.1, "经典推荐": 1.1},
        "金卡会员": {"进店必点": 1.3, "菌彩特色": 1.25, "经典推荐": 1.2, "云岭特色": 1.1, "山珍菌宴": 1.1},
        "钻石会员": {"进店必点": 1.5, "菌彩特色": 1.4, "经典推荐": 1.3, "云岭特色": 1.2, "山珍菌宴": 1.2, "鲜切牛肉": 1.1},
    }
    level_boost = membership_boost.get(membership_level, {})
    
    for category, dishes in category_pools.items():
        if not dishes:
            continue
            
        # 基础权重
        weight = base_weights.get(category, 1.0)
        
        # 根据人群类型调整权重
        if customer_type == "老人":
            if category in ["涮品", "山茅野菜"]:
                weight *= 0.7  # 老人少推荐涮菜和野菜
        elif customer_type == "儿童":
            if category in ["甜饮品"]:
                weight *= 1.3  # 儿童多推荐饮品
        elif customer_type == "聚餐":
            if category in ["进店必点", "经典推荐"]:
                weight *= 1.2  # 聚餐多推荐招牌菜
        
        # 根据天气调整权重
        if weather == "热天":
            if category in ["甜饮品"]:
                weight *= 1.3  # 热天多推荐饮品
            elif category in ["菌汤锅底"]:
                weight *= 0.8  # 热天少推荐热锅
        elif weather == "冷天":
            if category in ["菌汤锅底"]:
                weight *= 1.2  # 冷天多推荐热锅
        
        # 根据季节调整权重
        if season == "夏":
            if category in ["甜饮品", "涮品"]:
                weight *= 1.1
        elif season == "冬":
            if category in ["菌汤锅底", "普洱黄牛肉"]:
                weight *= 1.1
        
        # 会员等级加成
        if category in level_boost:
            weight *= level_boost[category]
        
        # 添加随机因子
        weight *= random.uniform(0.8, 1.2)
        
        weights[category] = max(0.1, weight)  # 确保权重不为负
    
    return weights


def _select_by_category_weights(category_pools, category_weights, recommended,
                              used_names, cat_count, max_per_category, remaining_slots,
                              total_quota, skipped_by_rules=None):
    """根据分类权重选择菜品（集成规则引擎冲突检测）"""
    import random

    if skipped_by_rules is None:
        skipped_by_rules = []

    # 创建候选池（未使用的菜品）
    available_candidates = []
    for category, dishes in category_pools.items():
        for dish in dishes:
            if dish.name not in used_names:
                # 添加权重信息
                weighted_dish = {
                    'dish': dish,
                    'category': category,
                    'weight': category_weights.get(category, 1.0)
                }
                available_candidates.append(weighted_dish)

    if not available_candidates:
        return

    # 按权重排序，但引入随机性
    available_candidates.sort(key=lambda x: x['weight'] * random.uniform(0.8, 1.2), reverse=True)

    # 选择菜品
    selected_count = 0
    for candidate in available_candidates:
        if selected_count >= remaining_slots:
            break
        # P0修复：检查是否已达总配额，防止超额推荐
        if len(recommended) >= total_quota:
            break

        dish = candidate['dish']
        category = candidate['category']

        # 检查分类限制
        if cat_count.get(category, 0) >= max_per_category:
            continue

        # 规则引擎冲突检测：跳过与已选菜品冲突的候选
        try:
            from dish_rules import has_conflict
        except ImportError:
            has_conflict = None
        if has_conflict is not None and has_conflict(dish.name, used_names):
            skipped_by_rules.append(dish.name)
            continue

        # 添加到推荐列表
        recommended.append(dish)
        used_names.add(dish.name)
        cat_count[category] = cat_count.get(category, 0) + 1
        selected_count += 1


def _select_from_remaining(category_pools, recommended, used_names, remaining_slots,
                          total_quota, skipped_by_rules=None):
    """从剩余菜品中随机选择（集成规则引擎冲突检测）"""
    import random

    if skipped_by_rules is None:
        skipped_by_rules = []

    # 收集所有剩余菜品
    remaining_dishes = []
    for dishes in category_pools.values():
        for dish in dishes:
            if dish.name not in used_names:
                remaining_dishes.append(dish)

    # 随机选择
    random.shuffle(remaining_dishes)

    # 添加到推荐列表
    for dish in remaining_dishes:
        # P0修复：使用传入的total_quota而非_total_quota(0)，原代码固定用5导致大组推荐不足
        if len(recommended) >= total_quota:
            break

        # 规则引擎冲突检测：跳过与已选菜品冲突的候选
        try:
            from dish_rules import has_conflict
        except ImportError:
            has_conflict = None
        if has_conflict is not None and has_conflict(dish.name, used_names):
            skipped_by_rules.append(dish.name)
            continue

        recommended.append(dish)
        used_names.add(dish.name)


def _build_recommend_context(taste, customer_type, weather, season,
                             allergen_avoid, people_count, membership_level,
                             recommended, skipped_by_rules):
    """构建结构化上下文（供 LLM 二次润色生成推荐理由，不面向顾客展示）"""
    ctx_parts = ["\n[推荐上下文]"]
    if people_count:
        ctx_parts.append(f"人数：{people_count}人")
    if taste:
        ctx_parts.append(f"口味：{taste}")
    if customer_type:
        ctx_parts.append(f"人群：{customer_type}")
    if weather:
        ctx_parts.append(f"天气：{weather}")
    if season:
        ctx_parts.append(f"季节：{season}")
    if membership_level and membership_level != "普通会员":
        ctx_parts.append(f"会员：{membership_level}")
    if allergen_avoid:
        ctx_parts.append(f"过敏避开：{allergen_avoid}")
    if skipped_by_rules:
        ctx_parts.append(f"规则避让：已跳过{len(skipped_by_rules)}道冲突菜品")
    # 品类分布
    if recommended:
        cats = list(set(d.category for d in recommended if d.category != "菌汤锅底"))
        if cats:
            ctx_parts.append(f"品类覆盖：{'、'.join(cats)}")
    return "\n".join(ctx_parts)


@tool
def generate_recommendation_reason(
    dish_names: str,
    taste: str = "",
    customer_type: str = "",
    weather: str = "",
    season: str = "",
    allergen_avoid: str = "",
    people_count: int = 0,
    membership_level: str = "",
    skipped_dishes: str = "",
) -> str:
    """为已推荐的菜品组合生成口语化、有人情味的推荐理由。

    适用场景：
    - 顾客问"为什么推荐这些菜？""说说这些菜好在哪？"
    - 需要在推荐结果之外，额外生成更详细的理由说明
    - 顾客想看推荐的"门道"

    后续计划：此工具可切换为 LLM 生成或本地知识库检索生成推荐理由。

    Args:
        dish_names: 推荐的菜品名称列表，多个用逗号分隔
        taste: 口味偏好
        customer_type: 人群类型
        weather: 天气
        season: 季节
        allergen_avoid: 已避开的过敏原
        people_count: 用餐人数
        membership_level: 会员等级
        skipped_dishes: 因规则冲突被跳过的菜品名称，多个用逗号分隔
    """
    import random

    dishes_list = [d.strip() for d in _parse_csv(dish_names) if d.strip()]
    skipped_list = [s.strip() for s in _parse_csv(skipped_dishes) if s.strip()]

    if not dishes_list:
        return "暂无推荐菜品，无法生成理由。"

    # 查找对应的 Dish 对象
    all_dishes = get_merged_dishes() if callable(get_merged_dishes) else []
    dish_map = {d.name: d for d in all_dishes}
    matched = [dish_map[name] for name in dishes_list if name in dish_map]

    parts = []

    # 开场：根据人数和会员等级
    if people_count and people_count > 0:
        if membership_level and membership_level != "普通会员":
            parts.append(f"作为{membership_level}，给{people_count}位客人精心挑选了这些好菜")
        elif people_count <= 2:
            parts.append(f"给{people_count}位客人挑了几道精致好菜")
        elif people_count <= 4:
            parts.append(f"给{people_count}位客人搭了一桌丰盛搭配")
        else:
            parts.append(f"给{people_count}位客人凑了一大桌")
    else:
        if membership_level and membership_level != "普通会员":
            parts.append(f"作为{membership_level}，为您精挑细选了这份搭配")
        else:
            parts.append("为您精挑细选了这份搭配")

    # 口味描述
    if taste:
        taste_words = {
            "不辣": "走的是清淡鲜美路线", "微辣": "带点微辣提提味刚刚好",
            "中辣": "香辣过瘾不会太刺激", "特辣": "辣劲十足吃起来特别爽",
            "酸辣": "酸辣开胃越吃越想吃", "香辣": "香气扑鼻很有层次",
        }
        display_taste = taste.split("/")[-1] if "/" in taste else taste
        parts.append(taste_words.get(display_taste, f"口味上{taste}"))

    # 搭配亮点（从菜品分类提取）
    if matched:
        categories = list(set(d.category for d in matched if d.category != "菌汤锅底"))
        if categories:
            cat_names = "、".join(categories[:3])
            parts.append(f"有{cat_names}等多种品类，搭配均衡")

    # 人群适配
    if customer_type:
        crowd_words = {
            "老人": "口味温和，都是长辈爱吃的",
            "儿童": "有好吃的也有营养的，小朋友肯定开心",
            "聚餐": "分量够足品类也全，大家一起分享才热闹",
            "情侣": "份量刚好不浪费，吃着也有情调",
            "一人食": "每道都精致，一个人也能吃出仪式感",
        }
        parts.append(crowd_words.get(customer_type, ""))

    # 天气时令点缀
    if weather or season:
        feel = ""
        if weather:
            weather_words = {"热天": "天热选了清爽的搭配", "冷天": "天冷来锅暖的", "雨天": "雨天吃火锅最治愈"}
            feel = weather_words.get(weather, "")
        if season:
            season_words = {"春": "春天就吃这一口鲜", "夏": "夏天也不会觉得腻", "秋": "秋天贴贴秋膘", "冬": "冬天暖身又暖心"}
            feel = season_words.get(season, feel)
        if feel:
            parts.append(feel)

    # 规则避让
    if skipped_list:
        parts.append(f"还特意帮您避开了{len(skipped_list)}道不太搭的菜，保证这桌不出错")

    # 过敏原
    if allergen_avoid:
        parts.append(f"「{allergen_avoid}」过敏的食材都帮您避开了，放心吃~")

    # 结尾
    parts.append("希望这桌菜能让您吃得开心！")

    # 过滤空值
    parts = [s for s in parts if s]
    return "。".join(parts)


# ======================== 加购工具（方案 A：统一入口 + LLM 路由） ========================
# 用户用自然语言说要加什么菜（如"来份水煮鱼加购"），LLM 识别意图后调用本工具。
# 工具内部从 contextvars 拿会话凭证，复用 CartAgent（LLM 提取菜名+数量 → 反查 → 加购）。
# 与 OrderingAgent 平行，但作为工具被 OrderingAgent 调用，实现统一入口。

@tool
def add_to_cart(user_input: str) -> str:
    """将顾客指定的菜品加入购物车。当顾客表达加购/下单意图时使用本工具。

    适用场景（必须调用本工具，不要口头回复"已加入"）：
    - 明确加购："来份XX加购""把XX加到购物车""加一份XX"
    - 确认下单："确认下单""就这些""下单吧"（基于最近推荐列表）
    - 自然语言加购："再来一份刚才那个XX""给我来两个XX"

    不适用场景（不要调用本工具）：
    - 纯推荐请求（"推荐几个菜"）→ 调用 recommend_dishes
    - 菜品咨询（"XX多少钱"）→ 调用 query_dish
    - 闲聊/打招呼 → 直接回复

    工具内部会从顾客的自然语言中提取菜名和数量，自动完成加购。
    如果顾客说"确认下单/就这些"但没指定菜名，工具会自动提取最近推荐的菜品。

    Args:
        user_input: 顾客的原始输入文本（如"来份水煮鱼加购""确认下单"）
    """
    # 从 contextvars 拿会话凭证（由 api_server.ai_chat 设置）
    ctx = _session_ctx.get()
    if ctx is None:
        return "加购失败：会话凭证缺失，请重新发起对话。"
    session_id, session_token = ctx

    # 复用 CartAgent 完成菜名提取 + 加购
    # CartAgent 通过模块级单例访问，避免每次调用重建 LLM 客户端
    try:
        from cart_agent import get_cart_agent
        agent = get_cart_agent()
    except Exception as e:
        logger.error("add_to_cart: CartAgent 未初始化: %s", e)
        return "加购服务暂时不可用，请稍后重试。"

    try:
        reply, cart_result = agent.chat(user_input, session_id, session_token)

        # 确认下单场景：CartAgent 未识别到具体菜名时，从会话历史提取最近推荐菜名
        # 用户说"确认下单/就这些"但没指定菜名，CartAgent 会返回"未识别"提示
        # 此时从 history 中最近一条推荐 AIMessage 提取菜名，重新加购
        if not cart_result and "未识别" in reply:
            history = _history_ctx.get()
            if history:
                dish_names = _extract_dish_names_from_history(history)
                if dish_names:
                    logger.info("add_to_cart: 确认下单场景，从历史提取 %d 道菜", len(dish_names))
                    # 把菜名拼接成 CartAgent 能识别的格式，重新调用
                    combined = "、".join(dish_names) + " 各一份加购"
                    reply, cart_result = agent.chat(combined, session_id, session_token)

        return reply
    except Exception as e:
        logger.exception("add_to_cart: CartAgent 调用失败")
        return f"加购失败：{e}"


# 所有工具列表（供 Agent 使用）
ALL_TOOLS = [
    query_dish,
    list_menu,
    recommend_dishes,
    # 菜品知识库工具（向量检索）
    search_dish_knowledge,
    get_pairing_plan,
    get_exclusion_rules,
    get_fruit_allergen_info,
    generate_recommendation_reason,
    # 加购工具（方案 A：统一入口 + LLM 路由）
    add_to_cart,
]

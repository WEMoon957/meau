"""工具函数模块 - 使用 LangChain @tool 装饰器定义工具

菜品查询、智能推荐、菜品知识库查询，供 LangChain Agent 调用。
"""

from langchain_core.tools import tool

from menu_data import Dish, get_all_dishes


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


def _filter_by_allergens(candidates: list[Dish], avoid_list: list[str]) -> list[Dish]:
    if not avoid_list:
        return candidates
    return [
        d for d in candidates
        if not any(allergen in d.allergens for allergen in avoid_list)
    ]


def _dish_score(d: Dish, weather_set: set[int], season_set: set[int]) -> int:
    score = 0
    if d.is_signature:
        score += 10
    if id(d) in weather_set:
        score += 5
    if id(d) in season_set:
        score += 3
    return score


def _category_quota(people_count: int) -> dict[str, int]:
    """根据人数返回各分类的推荐数量配额

    一人食：1热菜+1主食+1汤/饮品 = 3道
    2人食：1凉菜+2热菜+1主食+1汤 = 5道
    3-4人：2凉菜+3热菜+1汤+1主食+1饮品 = 8道
    5-8人：2凉菜+5热菜+1汤+2主食+1饮品+1甜点 = 12道
    8人以上：3凉菜+6热菜+2汤+2主食+2饮品+1甜点 = 16道
    """
    if people_count <= 0:
        return {"凉菜": 1, "热菜": 3, "汤品": 1, "主食": 1, "饮品": 1}
    if people_count <= 1:
        return {"热菜": 1, "主食": 1, "汤品": 1}
    if people_count <= 2:
        return {"凉菜": 1, "热菜": 2, "主食": 1, "汤品": 1}
    if people_count <= 4:
        return {"凉菜": 2, "热菜": 3, "汤品": 1, "主食": 1, "饮品": 1}
    if people_count <= 8:
        return {"凉菜": 2, "热菜": 5, "汤品": 1, "主食": 2, "饮品": 1, "甜点": 1}
    return {"凉菜": 3, "热菜": 6, "汤品": 2, "主食": 2, "饮品": 2, "甜点": 1}


# ======================== LangChain 工具定义 ========================

@tool
def query_dish(dish_name: str) -> str:
    """查询菜品详细信息，包括价格、辣度、适合人群、过敏原、饮食标签等。当顾客询问某道菜的具体信息时使用。

    使用参数化查询数据库，精确匹配优先，模糊匹配兜底。

    Args:
        dish_name: 菜品名称，如：宫保鸡丁、水煮鱼
    """
    from text_to_sql import query_dish_by_name
    return query_dish_by_name(dish_name)


@tool
def list_menu(category: str = "") -> str:
    """列出菜单菜品，可按分类筛选。当顾客想看菜单或浏览某类菜品时使用。

    使用参数化查询数据库，按分类白名单筛选。

    Args:
        category: 菜品分类：凉菜/热菜/汤品/主食/饮品/甜点，为空则列出全部
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
) -> str:
    """根据顾客需求智能推荐菜品。根据人数、口味、人群类型、健康标签、天气、季节、过敏原等多维度筛选推荐。推荐结果会自动搭配凉菜、热菜、汤品、主食、饮品。当顾客表达用餐需求或请求推荐时使用。

    Args:
        people_count: 用餐人数，0表示不限制
        taste: 口味偏好：不辣/微辣/中辣/特辣/酸辣/香辣/不限
        customer_type: 人群类型：儿童/老人/聚餐/情侣/一人食
        health_tags: 健康标签，多个用逗号分隔：低脂/低糖/高蛋白/素食/无麸质
        weather: 当前天气：热天/冷天/雨天
        season: 当前季节：春/夏/秋/冬
        allergen_avoid: 需要避开的过敏原，多个用逗号分隔：花生/海鲜/鸡蛋/牛奶/大豆
        include_drinks: 是否包含饮品，默认True。顾客明确要喝的时设为True
        include_staple: 是否包含主食，默认True
        include_soup: 是否包含汤品，默认True
    """
    all_dishes = get_all_dishes()
    candidates = list(all_dishes)
    avoid_list = _parse_csv(allergen_avoid)

    # 1. 口味筛选
    candidates = _filter_by_taste(candidates, taste)

    # 2. 人群筛选
    if customer_type:
        candidates = [d for d in candidates if customer_type in d.suitable_for]

    # 3. 健康标签筛选
    if health_tags:
        for tag in _parse_csv(health_tags):
            candidates = [d for d in candidates if tag in d.dietary_tags]

    # 4. 过敏原排除
    candidates = _filter_by_allergens(candidates, avoid_list)

    # 5/6. 天气/季节匹配（加分项）
    weather_set = {id(d) for d in candidates if weather and weather in d.weather_fit}
    season_set = {id(d) for d in candidates if season and season in d.seasonal}

    def sort_key(d: Dish) -> int:
        return -_dish_score(d, weather_set, season_set)

    candidates.sort(key=sort_key)

    # 按分类配额选取菜品（严格按人数控制总量，不再追加）
    quota = _category_quota(people_count)
    # 根据用户偏好调整配额
    if not include_drinks:
        quota.pop("饮品", None)
    if not include_staple:
        quota.pop("主食", None)
    if not include_soup:
        quota.pop("汤品", None)

    # 加载菜品规则引擎（互斥规则/避雷搭配）
    try:
        from dish_rules import has_conflict, get_rule_warnings
        rules_enabled = True
    except Exception:
        rules_enabled = False

    recommended: list[Dish] = []
    used_names: set[str] = set()
    skipped_by_rules: list[str] = []  # 因规则被跳过的菜品

    for cat, count in quota.items():
        selected_in_cat = 0
        # 优先从已筛选候选中选取
        cat_candidates = [d for d in candidates if d.category == cat and d.name not in used_names]
        for d in cat_candidates:
            if selected_in_cat >= count:
                break
            # 规则检查：跳过与已选菜品冲突的项
            if rules_enabled and has_conflict(d.name, used_names):
                skipped_by_rules.append(d.name)
                continue
            recommended.append(d)
            used_names.add(d.name)
            selected_in_cat += 1

        # 候选不足时从全菜单补充（仍受过敏原限制）
        if selected_in_cat < count:
            need = count - selected_in_cat
            pool = [d for d in all_dishes if d.category == cat and d.name not in used_names]
            pool = _filter_by_allergens(pool, avoid_list)
            pool.sort(key=sort_key)
            for d in pool:
                if need <= 0:
                    break
                # 规则检查
                if rules_enabled and has_conflict(d.name, used_names):
                    skipped_by_rules.append(d.name)
                    continue
                recommended.append(d)
                used_names.add(d.name)
                selected_in_cat += 1
                need -= 1

    # 按分类分组展示：菜名+价格在前，推荐理由放最后
    categories_order = ["凉菜", "热菜", "汤品", "主食", "饮品", "甜点"]
    lines = ["为您推荐以下菜品：\n"]

    total_price = 0.0
    idx = 1
    for cat in categories_order:
        cat_dishes = [d for d in recommended if d.category == cat]
        if cat_dishes:
            lines.append(f"--- {cat} ---")
            for d in cat_dishes:
                sig = " ★招牌" if d.is_signature else ""
                spicy = f" [{d.spicy_level}]" if d.spicy_level != "不辣" else ""
                lines.append(f"  {idx}. {d.name}  ￥{d.price}{spicy}{sig}")
                total_price += d.price
                idx += 1
            lines.append("")

    lines.append(f"合计：￥{total_price:.0f}")

    # 推荐理由放在最后
    reasons = []
    if taste:
        reasons.append(f"口味偏好「{taste}」")
    if customer_type:
        reasons.append(f"适合「{customer_type}」")
    if health_tags:
        reasons.append(f"健康需求「{health_tags}」")
    if weather:
        reasons.append(f"天气「{weather}」")
    if season:
        reasons.append(f"时令「{season}」")
    if allergen_avoid:
        reasons.append(f"已避开过敏原「{allergen_avoid}」")
    if reasons:
        lines.append(f"\n推荐理由：{'、'.join(reasons)}")

    # 规则合规验证：再次检查推荐结果无冲突
    if rules_enabled:
        rule_warnings = get_rule_warnings([d.name for d in recommended])
        if rule_warnings:
            lines.append(f"\n{rule_warnings}")
        if skipped_by_rules:
            lines.append(f"（已根据菜品互斥规则自动避开 {len(skipped_by_rules)} 道冲突菜品）")

    lines.append("\n如需调整推荐，请告诉我您的其他偏好！")
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
]

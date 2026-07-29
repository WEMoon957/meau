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
    # 归一化：儿童 → 小孩
    tag = customer_type.strip()
    if tag == "儿童":
        tag = "小孩"
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


def _generate_recommendation_explanation(
    recommended: list[Dish],
    taste: str = "",
    customer_type: str = "",
    weather: str = "",
    season: str = "",
    allergen_avoid: str = "",
    skipped_by_rules: list[str] = None,
) -> str:
    """基于规则引擎和向量库生成推荐理由

    结构：
      1. 用户偏好回显（口味/人群/天气/季节）
      2. 搭配关系说明（从向量库检索匹配的套餐方案）
      3. 规则合规说明（避开冲突菜品的数量和规则文本）
      4. 过敏原规避说明
    """
    reasons: list[str] = []

    # 1. 用户偏好回显
    prefs: list[str] = []
    if taste:
        prefs.append(f"口味偏好「{taste}」")
    if customer_type:
        prefs.append(f"适合「{customer_type}」")
    if weather:
        prefs.append(f"天气「{weather}」")
    if season:
        prefs.append(f"时令「{season}」")
    if prefs:
        reasons.append("、".join(prefs))

    # 2. 搭配关系说明（从向量库检索匹配的搭配方案）
    pairing_note = _find_pairing_evidence(recommended, customer_type, taste)
    if pairing_note:
        reasons.append(pairing_note)

    # 3. 规则合规说明
    if skipped_by_rules:
        # 从规则引擎取出被避开的具体规则文本，挑 1-2 条最具代表性的展示
        rule_detail = _extract_rule_detail(recommended, skipped_by_rules)
        if rule_detail:
            reasons.append(f"已自动避开 {len(skipped_by_rules)} 道冲突菜品（{rule_detail}）")
        else:
            reasons.append(f"已根据菜品互斥规则自动避开 {len(skipped_by_rules)} 道冲突菜品")

    # 4. 过敏原规避说明
    if allergen_avoid:
        reasons.append(f"已避开过敏原「{allergen_avoid}」")

    return "；".join(reasons) if reasons else ""


def _find_pairing_evidence(recommended: list[Dish], customer_type: str, taste: str) -> str:
    """从向量库检索匹配的搭配方案，生成搭配关系说明

    检索逻辑：
      - 用推荐菜品名 + 场景词组合成查询
      - 从 combo_plan 类型中检索 top1 相关方案
      - 若相关度 > 0.5，引用方案名称作为搭配依据
    """
    if not recommended:
        return ""

    try:
        from kb_query import search_combo_plans
    except Exception:
        return ""

    # 构造查询：用前几道菜名 + 场景词
    dish_names_sample = "、".join(d.name for d in recommended[:3])
    scene_word = customer_type or taste or ""
    query = f"{scene_word} {dish_names_sample} 搭配方案"

    try:
        results = search_combo_plans(query, top_k=1)
    except Exception:
        return ""

    if not results:
        return ""

    top = results[0]
    score = top.get("score", 0)
    # 相关度阈值：只引用相关度较高的方案
    if score < 0.5:
        return ""

    meta = top.get("metadata", {})
    section = meta.get("section", "")
    text = top.get("text", "")

    # 从方案文本中提取简短描述（排除 section 名本身，避免重复）
    brief = ""
    if text:
        # 取第一句
        first_sentence = text.split("。")[0].split("；")[0].split("\n")[0].strip()
        # 去掉与 section 重复的内容
        if first_sentence and first_sentence != section:
            brief = first_sentence[:50]

    if section:
        if brief:
            return f"参考「{section}」搭配方案（{brief}）"
        return f"参考「{section}」搭配方案"
    return ""


def _extract_rule_detail(recommended: list[Dish], skipped_by_rules: list[str]) -> str:
    """从规则引擎提取被避开菜品的具体规则说明

    返回简短的规则描述，如"菌子重复""口味冲突"等
    """
    try:
        from dish_rules import _get_rules
        engine = _get_rules()
    except Exception:
        return ""

    if not engine.rule_texts:
        return ""

    # 在规则文本中查找涉及被避开菜品的规则
    relevant_rules: list[str] = []
    skipped_set = set(skipped_by_rules)
    for rule in engine.rule_texts:
        # 规则涉及的菜品与被避开菜品有交集
        if set(rule["dishes"]) & skipped_set:
            section = rule.get("section", "")
            # 清理 section 名：去掉 emoji 和括号说明，保留核心
            if section:
                # 去掉常见 emoji 和装饰符号
                clean = section.replace("❌", "").replace("⚠️", "").strip()
                # 去掉括号及内容，如"避雷搭配（点菜避开这些组合）" → "避雷搭配"
                if "（" in clean:
                    clean = clean.split("（")[0].strip()
                if clean and clean not in relevant_rules:
                    relevant_rules.append(clean)
            if len(relevant_rules) >= 2:
                break

    if relevant_rules:
        return "、".join(relevant_rules[:2])

    return ""


# 菜名含这些关键词的菜品，仅在 customer_type 匹配时才纳入推荐
# 避免给成年人聚餐推荐儿童餐等场景不匹配的情况
_SPECIAL_DISH_KEYWORDS = {
    "儿童": {"儿童", "小孩"},
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
) -> str:
    """根据顾客需求智能推荐菜品。根据人数、口味、人群类型、健康标签、天气、季节、过敏原等多维度筛选并按综合评分排序推荐。评分包含毛利率权重（占 20%），优先推荐高毛利、招牌、应季菜品。当顾客表达用餐需求或请求推荐时使用。

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
        exclude_dishes: 需要排除的菜品名称（已推荐过的），多个用逗号分隔。用于追加推荐场景避免重复。锅底名称传入后仍会推荐（火锅店必选），但会尝试换一个锅底
    """
    all_dishes = get_all_dishes()
    candidates = list(all_dishes)
    avoid_list = _parse_csv(allergen_avoid)
    exclude_set = {x.strip() for x in _parse_csv(exclude_dishes) if x.strip()}

    # 1. 口味筛选
    candidates = _filter_by_taste(candidates, taste)

    # 2. 人群筛选（场景词不硬筛，只对真实人群标签筛选）
    candidates = _filter_by_customer_type(candidates, customer_type)

    # 2.5 场景过滤：儿童餐等特殊菜品仅在匹配场景下推荐
    candidates = _filter_by_scene(candidates, customer_type)

    # 3. 健康标签筛选
    if health_tags:
        for tag in _parse_csv(health_tags):
            candidates = [d for d in candidates if tag in d.dietary_tags]

    # 4. 过敏原排除
    candidates = _filter_by_allergens(candidates, avoid_list)

    # 4.5 排除已推荐过的菜品（追加场景去重）
    # 锅底不在此处排除（火锅店必选，单独处理换锅底逻辑）
    if exclude_set:
        candidates = [d for d in candidates
                      if d.name not in exclude_set
                      or (d.category == "菌汤锅底" and "锅" in d.name)]

    # 5. 品类开关过滤
    if not include_drinks:
        candidates = [d for d in candidates if d.category != "甜饮品"]
    if not include_soup:
        # 火锅店锅底必选：include_soup=False 仅过滤非锅底的汤类菜品
        # 菌汤锅底分类中名字含"锅"的为锅底（必选），其余为汤品（如山茅野菜拼盘，可过滤）
        candidates = [d for d in candidates
                      if d.category != "菌汤锅底" or "锅" in d.name]
    # 无论 include_soup 是否为 False，candidates 中菌汤锅底分类仅保留真锅底
    # （山茅野菜拼盘是涮菜拼盘，分类标注错误，不应出现在推荐候选中）
    candidates = [d for d in candidates
                  if d.category != "菌汤锅底" or "锅" in d.name]
    if not include_staple:
        # 主食类菜名关键词过滤（菌彩特色分类里包含主食）
        staple_keywords = ("饵丝", "饵块", "焖饭", "糯米粉", "米线", "面")
        candidates = [d for d in candidates
                      if not any(kw in d.name for kw in staple_keywords)]

    # 6. 天气/季节匹配（加分项）
    weather_set = {id(d) for d in candidates if weather and weather in d.weather_fit}
    season_set = {id(d) for d in candidates if season and season in d.seasonal}

    # 按评分排序（毛利率占 20%）
    candidates.sort(key=lambda d: -_dish_score(d, weather_set, season_set))

    # 加载菜品规则引擎（互斥规则/避雷搭配）
    try:
        from dish_rules import has_conflict, get_rule_warnings
        rules_enabled = True
    except Exception:
        rules_enabled = False

    # 按总数推荐，同分类最多 _MAX_PER_CATEGORY 道
    total_quota = _total_quota(people_count)
    recommended: list[Dish] = []
    used_names: set[str] = set()
    cat_count: dict[str, int] = {}
    skipped_by_rules: list[str] = []

    def _try_fill(pool: list[Dish], enforce_cat_limit: bool) -> None:
        """从 pool 中按评分顺序填充推荐列表"""
        for d in pool:
            if len(recommended) >= total_quota:
                break
            if d.name in used_names:
                continue
            if enforce_cat_limit and cat_count.get(d.category, 0) >= _MAX_PER_CATEGORY:
                continue
            if rules_enabled and has_conflict(d.name, used_names):
                skipped_by_rules.append(d.name)
                continue
            recommended.append(d)
            used_names.add(d.name)
            cat_count[d.category] = cat_count.get(d.category, 0) + 1

    # 火锅店必选锅底：至少 1 道菌汤锅底，不受口味/人群/健康标签筛选
    # 仅受过敏原限制；优先选名字含"锅"的真锅底，计入总配额，排在推荐列表首位
    # 蒜过敏兜底：若所有锅底均含过敏原，仍推荐唯一锅底并提示用户
    # 追加去重：若已推荐过锅底且存在其他锅底，则换一个不同的锅底
    pot_pool = _filter_by_allergens(
        [d for d in all_dishes if d.category == "菌汤锅底" and "锅" in d.name],
        avoid_list,
    )
    pot_warning = ""
    if not pot_pool:
        # 兜底：所有锅底均含过敏原，取唯一锅底并生成提示
        all_pots = [d for d in all_dishes if d.category == "菌汤锅底" and "锅" in d.name]
        if all_pots:
            pot_pool = all_pots
            pot_warning = f"⚠️ 提示：本店锅底「{all_pots[0].name}」含 {allergen_avoid or '过敏原'}，"
            if avoid_list:
                pot_warning += f"无法避开，请确认是否可接受；"
    if pot_pool:
        pot_pool.sort(key=lambda d: -_dish_score(d, weather_set, season_set))
        # 追加场景：优先选未排除的锅底；若全部已排除（只有一个锅底），仍推荐唯一锅底
        if exclude_set:
            fresh_pots = [d for d in pot_pool if d.name not in exclude_set]
            mandatory_pot = fresh_pots[0] if fresh_pots else pot_pool[0]
        else:
            mandatory_pot = pot_pool[0]
        recommended.append(mandatory_pot)
        used_names.add(mandatory_pot.name)
        cat_count[mandatory_pot.category] = 1

    # 第一轮：从筛选后候选中选，同分类限 2 道
    _try_fill(candidates, enforce_cat_limit=True)

    # 第二轮：候选不足时，放宽同分类限制（仍从筛选候选中选）
    if len(recommended) < total_quota:
        _try_fill(candidates, enforce_cat_limit=False)

    # 第三轮：仍不足时，从全菜单补充（放宽口味限制，仅保留过敏原过滤）
    if len(recommended) < total_quota:
        fallback_pool = _filter_by_allergens(all_dishes, avoid_list)
        fallback_pool.sort(key=lambda d: -_dish_score(d, weather_set, season_set))
        _try_fill(fallback_pool, enforce_cat_limit=False)

    # 按分类分组展示：菜名+价格在前，推荐理由放最后
    # 按分类在推荐中的出现顺序展示
    cat_order: list[str] = []
    for d in recommended:
        if d.category not in cat_order:
            cat_order.append(d.category)

    lines = ["为您推荐以下菜品：\n"]
    if pot_warning:
        lines.append(pot_warning + "\n")

    total_price = 0.0
    idx = 1
    for cat in cat_order:
        cat_dishes = [d for d in recommended if d.category == cat]
        if cat_dishes:
            lines.append(f"--- {cat} ---")
            for d in cat_dishes:
                sig = " ★招牌" if d.is_signature else ""
                spicy = f" [{d.spicy_level}]" if d.spicy_level and d.spicy_level != "不辣" else ""
                lines.append(f"  {idx}. {d.name}  ￥{d.price}{spicy}{sig}")
                total_price += d.price
                idx += 1
            lines.append("")

    lines.append(f"合计：￥{total_price:.0f}")

    # 推荐理由：基于规则引擎和向量库生成
    explanation = _generate_recommendation_explanation(
        recommended=recommended,
        taste=taste,
        customer_type=customer_type,
        weather=weather,
        season=season,
        allergen_avoid=allergen_avoid,
        skipped_by_rules=skipped_by_rules if skipped_by_rules else None,
    )
    if explanation:
        lines.append(f"\n推荐理由：{explanation}")

    # 规则合规验证：再次检查推荐结果无冲突
    if rules_enabled:
        rule_warnings = get_rule_warnings([d.name for d in recommended])
        if rule_warnings:
            lines.append(f"\n{rule_warnings}")

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

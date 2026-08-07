"""工具函数模块 - 使用 LangChain @tool 装饰器定义工具

菜品查询、智能推荐、购物车操作，供 LangChain Agent 调用。
"""

from contextvars import ContextVar

from langchain_core.tools import tool

from menu_data import Dish, find_dish_by_name, get_all_dishes


# ======================== 购物车 ========================
class Cart:
    """购物车管理"""

    def __init__(self):
        self._items: dict[str, dict] = {}  # {dish_name: {"dish": Dish, "quantity": int}}

    def add(self, dish: Dish, quantity: int = 1) -> str:
        if dish.name in self._items:
            self._items[dish.name]["quantity"] += quantity
        else:
            self._items[dish.name] = {"dish": dish, "quantity": quantity}
        return f"已将 {dish.name} x{quantity} 加入购物车"

    def remove(self, dish_name: str) -> str:
        if dish_name in self._items:
            del self._items[dish_name]
            return f"已将 {dish_name} 从购物车移除"
        return f"购物车中没有 {dish_name}"

    def update_quantity(self, dish_name: str, quantity: int) -> str:
        if dish_name not in self._items:
            return f"购物车中没有 {dish_name}"
        if quantity <= 0:
            return self.remove(dish_name)
        self._items[dish_name]["quantity"] = quantity
        return f"已将 {dish_name} 数量更新为 {quantity}"

    def clear(self) -> str:
        self._items.clear()
        return "购物车已清空"

    def get_summary(self) -> str:
        if not self._items:
            return "购物车是空的"
        lines = ["========== 购物车 =========="]
        total = 0.0
        for item in self._items.values():
            dish = item["dish"]
            qty = item["quantity"]
            subtotal = dish.price * qty
            total += subtotal
            lines.append(f"  {dish.name} x{qty}  ￥{dish.price} x {qty} = ￥{subtotal:.0f}")
        lines.append(f"----------------------------")
        lines.append(f"  合计: ￥{total:.0f}")
        lines.append("============================")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self._items) == 0


_session_id: ContextVar[str] = ContextVar("session_id", default="default")
_carts: dict[str, Cart] = {}


def set_session_id(session_id: str) -> None:
    """设置当前请求的会话 ID（API 多用户隔离）"""
    _session_id.set(session_id or "default")


def get_session_id() -> str:
    return _session_id.get()


def _get_session_cart() -> Cart:
    """获取当前会话的购物车"""
    sid = _session_id.get()
    if sid not in _carts:
        _carts[sid] = Cart()
    return _carts[sid]


def reset_cart(session_id: str | None = None) -> None:
    """清空指定会话的购物车"""
    sid = session_id or _session_id.get()
    if sid in _carts:
        _carts[sid].clear()


def destroy_session(session_id: str) -> None:
    """彻底移除会话购物车（会话过期时调用）"""
    _carts.pop(session_id or "default", None)


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
    """根据人数返回各分类的推荐数量配额（与系统提示词策略一致）

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
async def query_dish(dish_name: str) -> str:
    """查询菜品详细信息，包括价格、辣度、适合人群、过敏原、饮食标签等。当顾客询问某道菜的具体信息时使用。

    使用 Text-to-SQL 技术：LLM 根据菜品名称生成 SQL 查询数据库，精确匹配优先，模糊匹配兜底。

    Args:
        dish_name: 菜品名称，如：宫保鸡丁、水煮鱼
    """
    from text_to_sql import aquery_dish_by_name
    return await aquery_dish_by_name(dish_name)


@tool
async def list_menu(category: str = "") -> str:
    """列出菜单菜品，可按分类筛选。当顾客想看菜单或浏览某类菜品时使用。

    使用 Text-to-SQL 技术：LLM 根据分类生成 SQL 查询数据库。

    Args:
        category: 菜品分类：凉菜/热菜/汤品/主食/饮品/甜点，为空则列出全部
    """
    from text_to_sql import alist_menu_dishes
    return await alist_menu_dishes(category)


@tool
async def recommend_dishes(
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

    recommended: list[Dish] = []
    used_names: set[str] = set()

    for cat, count in quota.items():
        # 优先从已筛选候选中选取
        cat_candidates = [d for d in candidates if d.category == cat and d.name not in used_names]
        for d in cat_candidates[:count]:
            recommended.append(d)
            used_names.add(d.name)
        # 候选不足时从全菜单补充（仍受过敏原限制）
        if len([d for d in recommended if d.category == cat]) < count:
            need = count - len([d for d in recommended if d.category == cat])
            pool = [d for d in all_dishes if d.category == cat and d.name not in used_names]
            pool = _filter_by_allergens(pool, avoid_list)
            pool.sort(key=sort_key)
            for d in pool[:need]:
                recommended.append(d)
                used_names.add(d.name)

    # 按分类分组展示，带序号和合计
    categories_order = ["凉菜", "热菜", "汤品", "主食", "饮品", "甜点"]
    lines = ["根据您的需求，为您推荐以下菜品：\n"]
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
        lines.append(f"推荐依据：{'、'.join(reasons)}\n")

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
                lines.append(f"    {d.description}")
                total_price += d.price
                idx += 1
            lines.append("")

    lines.append(f"合计：￥{total_price:.0f}")
    lines.append("\n如需将推荐菜品加入购物车，请告诉我！")
    return "\n".join(lines)


@tool
async def add_to_cart(dish_name: str, quantity: int = 1) -> str:
    """将菜品加入购物车。当顾客确认要点某道菜时使用。

    Args:
        dish_name: 菜品名称
        quantity: 数量，默认1
    """
    dish = find_dish_by_name(dish_name)
    if not dish:
        return f"未找到菜品「{dish_name}」，请确认菜品名称。"
    return _get_session_cart().add(dish, quantity)


@tool
async def quick_add_from_category(category: str) -> str:
    """从指定分类中直接选一道菜加入购物车。当顾客说"来个喝的就行""随便来个汤""来个主食"等不需要指定具体菜品时使用。

    Args:
        category: 菜品分类：凉菜/热菜/汤品/主食/饮品/甜点
    """
    import random
    all_dishes = get_all_dishes()
    dishes = [d for d in all_dishes if d.category == category]
    if not dishes:
        return f"没有找到分类「{category}」的菜品，可选分类：凉菜/热菜/汤品/主食/饮品/甜点"
    # 优先选招牌菜
    signatures = [d for d in dishes if d.is_signature]
    pick = random.choice(signatures if signatures else dishes)
    _get_session_cart().add(pick, 1)
    return f"已为您选了「{pick.name}」（￥{pick.price}）加入购物车。{pick.description}"


@tool
async def remove_from_cart(dish_name: str) -> str:
    """从购物车移除菜品。当顾客想取消某道菜时使用。

    Args:
        dish_name: 菜品名称
    """
    return _get_session_cart().remove(dish_name)


@tool
async def get_cart() -> str:
    """查看当前购物车内容和总价。当顾客想查看已选菜品时使用。"""
    return _get_session_cart().get_summary()


@tool
async def checkout() -> str:
    """结算下单。当顾客确认要下单时使用，会清空购物车。"""
    session_cart = _get_session_cart()
    if session_cart.is_empty():
        return "购物车是空的，无法下单。请先添加菜品。"
    summary = session_cart.get_summary()
    session_cart.clear()
    return f"下单成功！\n\n{summary}\n\n感谢您的光临，菜品将尽快为您准备！"


# ======================== 服务员话术工具 ========================

@tool
async def generate_server_script(scene: str, script_type: str = "") -> str:
    """根据场景从话术向量库中检索并生成服务员话术。必须在此类场景下使用本工具，禁止自行编造回答：
    - 顾客嫌菜太辣/太咸/不好吃等客诉应对
    - 顾客带小孩/老人来用餐的推荐
    - 如何迎宾、如何推荐招牌菜、如何结账
    - 顾客对某些食材过敏的提醒
    - 菜品缺货怎么应对
    - 顾客等太久怎么安抚
    - 任何"怎么应对/怎么说/怎么推荐"的服务沟通问题

    Args:
        scene: 场景描述，如"顾客带小孩来用餐""顾客嫌菜太辣""推荐招牌菜""四个人聚餐怎么推荐"
        script_type: 话术类型筛选（可选）：selling_point(菜品卖点)/scene(场景应对)/pairing(搭配推荐)/exception(异常处理)
    """
    from vector_store import asearch_scripts

    results = await asearch_scripts(scene, k=3, script_type=script_type if script_type else None)

    if not results:
        return "暂未找到匹配的话术，请尝试换个描述。"

    lines = [f"为您找到 {len(results)} 条相关话术：\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[话术{i}] (相关度: {r['score']})")
        lines.append(r["content"])
        lines.append("")

    return "\n".join(lines).strip()


@tool
async def add_custom_script(content: str, script_type: str = "custom", dish_name: str = "", scene: str = "") -> str:
    """添加自定义话术到话术向量库，供后续检索使用。当服务员或店长想补充新的话术时使用。

    Args:
        content: 话术内容
        script_type: 话术类型：selling_point/scene/pairing/exception/custom，默认custom
        dish_name: 关联的菜品名称（可选）
        scene: 场景描述（可选）
    """
    from vector_store import aadd_script

    return await aadd_script(content, script_type, dish_name, scene)


# 所有工具列表（供 Agent 使用）
ALL_TOOLS = [
    query_dish,
    list_menu,
    recommend_dishes,
    add_to_cart,
    quick_add_from_category,
    remove_from_cart,
    get_cart,
    checkout,
    generate_server_script,
    add_custom_script,
]

"""菜单数据模块 - 提供菜品数据模型、初始数据和检索能力

数据模型和初始数据定义在此模块中。
运行时通过 db.py 从MySQL数据库加载菜品数据。
初始化数据库时，init_db.py 会将 MENU 列表导入到数据库。
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Dish:
    """菜品数据模型"""
    id: int
    name: str
    price: float
    category: str  # 凉菜/热菜/汤品/主食/饮品/甜点
    spicy_level: str  # 不辣/微辣/中辣/特辣
    suitable_for: list[str]  # 适合人群：儿童/老人/聚餐/情侣/一人食
    dietary_tags: list[str]  # 素食/清真/低脂/低糖/高蛋白/无麸质
    allergens: list[str]  # 过敏原：花生/海鲜/鸡蛋/牛奶/大豆
    description: str
    is_signature: bool = False  # 是否招牌菜/高毛利菜
    seasonal: list[str] = field(default_factory=list)  # 适合季节：春/夏/秋/冬
    weather_fit: list[str] = field(default_factory=list)  # 适合天气：热天/冷天/雨天


# ======================== 菜品数据库 ========================
MENU: list[Dish] = [
    # ---------- 凉菜 ----------
    Dish(
        id=1, name="口水鸡", price=38, category="凉菜", spicy_level="中辣",
        suitable_for=["聚餐", "情侣"], dietary_tags=["高蛋白"],
        allergens=["花生"], description="经典川式凉菜，鸡肉嫩滑配麻辣红油，开胃下饭",
        is_signature=True, seasonal=["夏", "春"], weather_fit=["热天"],
    ),
    Dish(
        id=2, name="凉拌木耳", price=18, category="凉菜", spicy_level="微辣",
        suitable_for=["聚餐", "一人食", "老人"], dietary_tags=["素食", "低脂"],
        allergens=[], description="爽脆木耳配香醋蒜泥，清脆爽口",
        seasonal=["夏", "春"], weather_fit=["热天"],
    ),
    Dish(
        id=3, name="老醋花生", price=16, category="凉菜", spicy_level="不辣",
        suitable_for=["聚餐", "下酒"], dietary_tags=["素食"],
        allergens=["花生"], description="油炸花生配老醋洋葱，酥脆酸甜",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=4, name="蒜泥白肉", price=32, category="凉菜", spicy_level="中辣",
        suitable_for=["聚餐"], dietary_tags=["高蛋白"],
        allergens=[], description="薄切五花肉配蒜泥红油，肥而不腻",
        seasonal=[], weather_fit=[],
    ),

    # ---------- 热菜 ----------
    Dish(
        id=5, name="宫保鸡丁", price=42, category="热菜", spicy_level="微辣",
        suitable_for=["聚餐", "儿童", "一人食"], dietary_tags=["高蛋白"],
        allergens=["花生"], description="经典川菜，鸡丁滑嫩配花生米，甜辣适口",
        is_signature=True, seasonal=[], weather_fit=[],
    ),
    Dish(
        id=6, name="水煮鱼", price=68, category="热菜", spicy_level="特辣",
        suitable_for=["聚餐"], dietary_tags=["高蛋白"],
        allergens=[], description="鲜嫩鱼片浸在红油辣椒中，麻辣鲜香",
        is_signature=True, seasonal=["冬", "秋"], weather_fit=["冷天", "雨天"],
    ),
    Dish(
        id=7, name="番茄炒蛋", price=22, category="热菜", spicy_level="不辣",
        suitable_for=["儿童", "老人", "一人食", "情侣"], dietary_tags=["素食", "高蛋白"],
        allergens=[], description="家常经典，酸甜番茄配嫩滑鸡蛋，老少皆宜",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=8, name="红烧排骨", price=48, category="热菜", spicy_level="不辣",
        suitable_for=["聚餐", "儿童", "老人"], dietary_tags=["高蛋白"],
        allergens=[], description="排骨酥烂入味，酱香浓郁，下饭神器",
        is_signature=True, seasonal=["冬", "秋"], weather_fit=["冷天"],
    ),
    Dish(
        id=9, name="清蒸鲈鱼", price=78, category="热菜", spicy_level="不辣",
        suitable_for=["聚餐", "儿童", "老人"], dietary_tags=["高蛋白", "低脂"],
        allergens=[], description="新鲜鲈鱼清蒸，肉质细嫩，原汁原味",
        is_signature=True, seasonal=["春", "秋"], weather_fit=[],
    ),
    Dish(
        id=10, name="麻婆豆腐", price=28, category="热菜", spicy_level="中辣",
        suitable_for=["聚餐", "一人食"], dietary_tags=["素食"],
        allergens=[], description="嫩豆腐配麻辣肉末，麻辣烫鲜，下饭首选",
        seasonal=["冬"], weather_fit=["冷天", "雨天"],
    ),
    Dish(
        id=11, name="干煸四季豆", price=26, category="热菜", spicy_level="微辣",
        suitable_for=["聚餐", "一人食"], dietary_tags=["素食", "低脂"],
        allergens=[], description="四季豆干煸至表皮微皱，配肉末煸炒，香脆可口",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=12, name="糖醋里脊", price=38, category="热菜", spicy_level="不辣",
        suitable_for=["儿童", "聚餐", "情侣"], dietary_tags=["高蛋白"],
        allergens=[], description="外酥里嫩的猪里脊裹糖醋汁，酸甜开胃",
        is_signature=True, seasonal=[], weather_fit=[],
    ),
    Dish(
        id=13, name="蒜蓉西兰花", price=20, category="热菜", spicy_level="不辣",
        suitable_for=["儿童", "老人", "一人食"], dietary_tags=["素食", "低脂", "低糖"],
        allergens=[], description="新鲜西兰花配蒜蓉清炒，清淡健康",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=14, name="小炒黄牛肉", price=52, category="热菜", spicy_level="中辣",
        suitable_for=["聚餐"], dietary_tags=["高蛋白"],
        allergens=[], description="黄牛肉配泡椒芹菜爆炒，鲜嫩香辣",
        seasonal=["冬", "秋"], weather_fit=["冷天"],
    ),

    # ---------- 汤品 ----------
    Dish(
        id=15, name="番茄蛋花汤", price=12, category="汤品", spicy_level="不辣",
        suitable_for=["儿童", "老人", "一人食", "聚餐"], dietary_tags=["素食", "高蛋白"],
        allergens=[], description="番茄鸡蛋汤，酸甜暖胃",
        seasonal=["冬"], weather_fit=["冷天", "雨天"],
    ),
    Dish(
        id=16, name="酸萝卜老鸭汤", price=48, category="汤品", spicy_level="不辣",
        suitable_for=["聚餐", "老人"], dietary_tags=["高蛋白"],
        allergens=[], description="老鸭炖汤配酸萝卜，开胃滋补",
        is_signature=True, seasonal=["冬", "秋"], weather_fit=["冷天", "雨天"],
    ),
    Dish(
        id=17, name="紫菜蛋花汤", price=10, category="汤品", spicy_level="不辣",
        suitable_for=["儿童", "老人", "一人食"], dietary_tags=["素食", "高蛋白"],
        allergens=[], description="紫菜鸡蛋快速汤，清淡鲜美",
        seasonal=[], weather_fit=[],
    ),

    # ---------- 主食 ----------
    Dish(
        id=18, name="蛋炒饭", price=15, category="主食", spicy_level="不辣",
        suitable_for=["一人食", "儿童", "老人"], dietary_tags=["高蛋白"],
        allergens=[], description="蛋香米饭粒粒分明，简单美味",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=19, name="担担面", price=18, category="主食", spicy_level="中辣",
        suitable_for=["一人食", "情侣"], dietary_tags=[],
        allergens=["花生"], description="麻辣鲜香的面条配肉末，川味十足",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=20, name="葱油拌面", price=14, category="主食", spicy_level="不辣",
        suitable_for=["一人食", "儿童"], dietary_tags=["素食"],
        allergens=[], description="葱油酱油拌面，简单鲜香",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=21, name="扬州炒饭", price=22, category="主食", spicy_level="不辣",
        suitable_for=["聚餐", "一人食", "儿童"], dietary_tags=["高蛋白"],
        allergens=[], description="什锦炒饭配虾仁火腿，料足味美",
        seasonal=[], weather_fit=[],
    ),

    # ---------- 饮品 ----------
    Dish(
        id=22, name="酸梅汤", price=8, category="饮品", spicy_level="不辣",
        suitable_for=["聚餐", "儿童", "老人"], dietary_tags=["低脂"],
        allergens=[], description="酸甜解腻的传统饮品，冰镇更佳",
        seasonal=["夏"], weather_fit=["热天"],
    ),
    Dish(
        id=23, name="柠檬蜂蜜水", price=12, category="饮品", spicy_level="不辣",
        suitable_for=["儿童", "老人", "一人食"], dietary_tags=["低脂"],
        allergens=[], description="柠檬蜂蜜温水，清新润喉",
        seasonal=["夏", "春"], weather_fit=["热天"],
    ),
    Dish(
        id=24, name="热豆浆", price=6, category="饮品", spicy_level="不辣",
        suitable_for=["儿童", "老人", "一人食"], dietary_tags=["素食", "低糖"],
        allergens=["大豆"], description="现磨热豆浆，暖胃营养",
        seasonal=["冬"], weather_fit=["冷天"],
    ),

    # ---------- 甜点 ----------
    Dish(
        id=25, name="红豆双皮奶", price=16, category="甜点", spicy_level="不辣",
        suitable_for=["儿童", "情侣", "一人食"], dietary_tags=["低脂"],
        allergens=["牛奶"], description="顺德双皮奶配蜜红豆，嫩滑香甜",
        seasonal=[], weather_fit=[],
    ),
    Dish(
        id=26, name="芒果布丁", price=14, category="甜点", spicy_level="不辣",
        suitable_for=["儿童", "情侣"], dietary_tags=[],
        allergens=["牛奶"], description="新鲜芒果制布丁，Q弹爽滑",
        seasonal=["夏"], weather_fit=["热天"],
    ),
    Dish(
        id=27, name="桂花糕", price=12, category="甜点", spicy_level="不辣",
        suitable_for=["老人", "聚餐", "一人食"], dietary_tags=["素食"],
        allergens=[], description="传统桂花糕，软糯清香",
        seasonal=["秋"], weather_fit=[],
    ),
]


def _use_database() -> bool:
    """检查是否启用数据库模式"""
    return os.environ.get("USE_DATABASE", "true").lower() == "true"


_dishes_cache: list[Dish] | None = None
_dish_name_index: dict[str, Dish] | None = None


def _build_name_index(dishes: list[Dish]) -> dict[str, Dish]:
    """构建菜名索引，加速模糊查找"""
    index: dict[str, Dish] = {}
    for dish in dishes:
        index[dish.name] = dish
    return index


def _load_dishes() -> list[Dish]:
    """从数据库或本地加载菜品（仅首次调用时执行 I/O）"""
    if _use_database():
        try:
            from db import fetch_all_dishes
            return fetch_all_dishes()
        except Exception as e:
            print(f"⚠️ 数据库读取失败，回退到本地数据: {e}")
    return MENU


def invalidate_dishes_cache() -> None:
    """清除菜品缓存（数据库更新后调用）"""
    global _dishes_cache, _dish_name_index
    _dishes_cache = None
    _dish_name_index = None


def get_all_dishes() -> list[Dish]:
    """获取所有菜品（内存缓存，避免重复数据库查询）"""
    global _dishes_cache, _dish_name_index
    if _dishes_cache is None:
        _dishes_cache = _load_dishes()
        _dish_name_index = _build_name_index(_dishes_cache)
    return _dishes_cache


def find_dish_by_name(name: str) -> Optional[Dish]:
    """根据名称查找菜品（模糊匹配，使用内存索引）"""
    name = name.strip()
    if not name:
        return None

    dishes = get_all_dishes()
    index = _dish_name_index or {}

    if name in index:
        return index[name]

    for dish_name, dish in index.items():
        if name in dish_name or dish_name in name:
            return dish
    return None


def find_dishes_by_category(category: str) -> list[Dish]:
    """根据分类查找菜品"""
    return [d for d in get_all_dishes() if d.category == category]


def format_dish_info(dish: Dish) -> str:
    """格式化菜品信息为可读文本"""
    spicy = f"【{dish.spicy_level}】" if dish.spicy_level != "不辣" else "【不辣】"
    signature = " ★招牌" if dish.is_signature else ""
    allergen_text = f"过敏原: {','.join(dish.allergens)}" if dish.allergens else "无常见过敏原"
    dietary_text = f"饮食标签: {','.join(dish.dietary_tags)}" if dish.dietary_tags else "无特殊饮食标签"
    suitable_text = f"适合: {','.join(dish.suitable_for)}" if dish.suitable_for else "通用"

    return (
        f"【{dish.name}】 ￥{dish.price} ({dish.category}){signature}\n"
        f"  辣度: {spicy}\n"
        f"  {suitable_text}\n"
        f"  {dietary_text}\n"
        f"  {allergen_text}\n"
        f"  介绍: {dish.description}"
    )

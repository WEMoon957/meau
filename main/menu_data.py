"""菜单数据模块 - 提供菜品数据模型、初始数据和检索能力

数据模型和初始数据定义在此模块中。
运行时通过 db.py 从MySQL数据库加载菜品数据。
初始化数据库时，init_db.py 会将 MENU 列表导入到数据库。
"""

from dataclasses import dataclass, field
from typing import Optional
import os


# ======================== 数据模型 ========================
@dataclass
class Dish:
    """菜品数据模型"""
    id: int
    name: str
    price: float
    category: str
    spicy_level: str
    suitable_for: list = field(default_factory=list)
    dietary_tags: list = field(default_factory=list)
    allergens: list = field(default_factory=list)
    description: str = ""
    is_signature: bool = False
    seasonal: list = field(default_factory=list)
    weather_fit: list = field(default_factory=list)
    gross_margin: float = 0.0  # 毛利率 0~1，推荐评分权重占 20%


# ======================== 初始菜品数据（用于首次初始化数据库） ========================
MENU: list[Dish] = [
    Dish(1, "口水鸡", 38, "凉菜", "中辣",
         ["聚餐", "情侣"], ["高蛋白"], ["花生"],
         "经典川式凉菜，鸡肉嫩滑配麻辣红油，开胃下饭", True,
         ["夏", "春"], ["热天"]),
    Dish(2, "凉拌木耳", 18, "凉菜", "微辣",
         ["聚餐", "一人食", "老人"], ["素食", "低脂"], [],
         "爽脆木耳配香醋蒜泥，清脆爽口", False, [], []),
    Dish(3, "老醋花生", 16, "凉菜", "不辣",
         ["下酒"], [], ["花生"],
         "油炸花生配老醋洋葱，酥脆酸甜", False, [], []),
    Dish(4, "蒜泥白肉", 32, "凉菜", "中辣",
         ["聚餐"], [], [],
         "薄切五花肉配蒜泥红油，肥而不腻", False, [], []),
    Dish(5, "宫保鸡丁", 42, "热菜", "中辣",
         ["聚餐", "儿童", "一人食"], [], ["花生"],
         "经典川菜，鸡丁滑嫩配花生米，甜辣适口", False, [], []),
    Dish(6, "水煮鱼", 68, "热菜", "特辣",
         ["聚餐"], [], [],
         "鲜嫩鱼片浸在红油辣椒中，麻辣鲜香", False,
         ["冬", "秋"], ["冷天", "雨天"]),
    Dish(7, "番茄炒蛋", 22, "热菜", "不辣",
         ["儿童", "老人", "一人食", "情侣"], [], [],
         "家常经典，酸甜番茄配嫩滑鸡蛋，老少皆宜", False, [], []),
    Dish(8, "红烧排骨", 48, "热菜", "不辣",
         ["聚餐", "儿童", "老人"], [], [],
         "排骨酥烂入味，酱香浓郁，下饭神器", False, [], []),
    Dish(9, "清蒸鲈鱼", 78, "热菜", "不辣",
         ["聚餐"], [], [],
         "新鲜鲈鱼清蒸，肉质细嫩，原汁原味", False, [], []),
    Dish(10, "麻婆豆腐", 28, "热菜", "中辣",
         ["一人食"], [], [],
         "嫩豆腐配麻辣肉末，麻辣烫鲜，下饭首选", False, [], []),
    Dish(11, "干煸四季豆", 26, "热菜", "微辣",
         ["聚餐"], [], [],
         "四季豆干煸至表皮微皱，配肉末煸炒，香脆可口", False, [], []),
    Dish(12, "糖醋里脊", 36, "热菜", "不辣",
         ["儿童", "聚餐", "情侣"], [], [],
         "外酥里嫩的猪里脊裹糖醋汁，酸甜开胃", False, [], []),
    Dish(13, "蒜蓉西兰花", 20, "热菜", "不辣",
         ["儿童", "老人", "一人食"], ["素食", "低脂", "低糖"], [],
         "新鲜西兰花配蒜蓉清炒，清淡健康", False, [], []),
    Dish(14, "小炒黄牛肉", 52, "热菜", "中辣",
         ["聚餐"], [], [],
         "黄牛肉配泡椒芹菜爆炒，鲜嫩香辣", False, [], []),
    Dish(15, "番茄蛋花汤", 18, "汤品", "不辣",
         ["儿童", "老人", "一人食", "聚餐"], [], [],
         "番茄鸡蛋汤，酸甜暖胃", False, [], []),
    Dish(16, "酸萝卜老鸭汤", 38, "汤品", "不辣",
         ["老人"], [], [],
         "老鸭炖汤配酸萝卜，开胃滋补", False, [], []),
    Dish(17, "紫菜蛋花汤", 12, "汤品", "不辣",
         ["一人食"], [], [],
         "紫菜鸡蛋快速汤，清淡鲜美", False, [], []),
    Dish(18, "蛋炒饭", 16, "主食", "不辣",
         ["一人食", "儿童", "老人"], [], [],
         "蛋香米饭粒粒分明，简单美味", False, [], []),
    Dish(19, "担担面", 22, "主食", "微辣",
         ["一人食"], [], [],
         "麻辣鲜香的面条配肉末，川味十足", False, [], []),
    Dish(20, "葱油拌面", 18, "主食", "不辣",
         ["儿童"], [], [],
         "葱油酱油拌面，简单鲜香", False, [], []),
    Dish(21, "扬州炒饭", 28, "主食", "不辣",
         ["聚餐", "一人食", "儿童"], [], [],
         "什锦炒饭配虾仁火腿，料足味美", False, [], []),
    Dish(22, "酸梅汤", 12, "饮品", "不辣",
         ["聚餐"], [], [],
         "酸甜解腻的传统饮品，冰镇更佳", False,
         ["夏"], ["热天"]),
    Dish(23, "柠檬蜂蜜水", 15, "饮品", "不辣",
         [], [], [],
         "柠檬蜂蜜温水，清新润喉", False, [], []),
    Dish(24, "热豆浆", 8, "饮品", "不辣",
         [], ["低糖"], ["大豆"],
         "现磨热豆浆，暖胃营养", False,
         ["冬"], ["冷天"]),
    Dish(25, "红豆双皮奶", 18, "甜点", "不辣",
         ["儿童", "情侣", "一人食"], [], ["牛奶"],
         "顺德双皮奶配蜜红豆，嫩滑香甜", False, [], []),
    Dish(26, "芒果布丁", 16, "甜点", "不辣",
         ["儿童"], [], [],
         "新鲜芒果制布丁，Q弹爽滑", False, [], []),
    Dish(27, "桂花糕", 15, "甜点", "不辣",
         ["老人", "聚餐", "一人食"], [], [],
         "传统桂花糕，软糯清香", False, [], []),
]


# ======================== 运行时缓存 ========================
_dishes_cache: Optional[list[Dish]] = None
_dish_name_index: dict[str, Dish] = {}


def _use_database() -> bool:
    """检查是否启用数据库模式"""
    return os.environ.get("USE_DATABASE", "true").lower() == "true"


def _build_name_index(dishes: list[Dish]):
    """构建菜名索引，加速模糊查找"""
    global _dish_name_index
    _dish_name_index = {d.name: d for d in dishes}


def _load_dishes():
    """从数据库或本地加载菜品（仅首次调用时执行 I/O）"""
    global _dishes_cache
    if _use_database():
        try:
            import db
            _dishes_cache = db.fetch_all_dishes()
        except Exception as e:
            print(f"⚠️ 数据库读取失败，回退到本地数据: {e}")
            _dishes_cache = MENU
    else:
        _dishes_cache = MENU


def invalidate_dishes_cache():
    """清除菜品缓存（数据库更新后调用）"""
    global _dishes_cache, _dish_name_index
    _dishes_cache = None
    _dish_name_index = {}


def get_all_dishes() -> list[Dish]:
    """获取所有菜品（内存缓存，避免重复数据库查询）"""
    global _dishes_cache, _dish_name_index
    if _dishes_cache is None:
        _load_dishes()
    if not _dish_name_index:
        _build_name_index(_dishes_cache)
    return _dishes_cache


def find_dish_by_name(name: str) -> Optional[Dish]:
    """根据名称查找菜品（模糊匹配，使用内存索引）"""
    name = name.strip()
    dishes = get_all_dishes()
    # 精确匹配
    if name in _dish_name_index:
        return _dish_name_index[name]
    # 模糊匹配
    for dish_name, dish in _dish_name_index.items():
        if name in dish_name or dish_name in name:
            return dish
    return None


def find_dishes_by_category(category: str) -> list[Dish]:
    """根据分类查找菜品"""
    dishes = get_all_dishes()
    return [d for d in dishes if d.category == category]


def format_dish_info(dish: Dish) -> str:
    """格式化菜品信息为可读文本"""
    spicy = f"【{dish.spicy_level}】" if dish.spicy_level != "不辣" else "【不辣】"
    sig = " ★招牌" if dish.is_signature else ""

    allergen_text = ", ".join(dish.allergens) if dish.allergens else "无常见过敏原"
    diet_text = ", ".join(dish.dietary_tags) if dish.dietary_tags else "无特殊饮食标签"
    crowd_text = ", ".join(dish.suitable_for) if dish.suitable_for else "通用"

    lines = [
        f"{spicy}{dish.name}{sig} ￥{dish.price} ({dish.category})",
        f"  辣度: {dish.spicy_level}",
        f"  过敏原: {allergen_text}",
        f"  饮食标签: {diet_text}",
        f"  适合: {crowd_text}",
        f"  介绍: {dish.description}",
    ]
    return "\n".join(lines)

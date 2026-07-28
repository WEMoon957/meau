"""数据库连接模块 - MySQL连接管理与菜品数据CRUD（使用pymysql）"""

import os
import json
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional

from menu_data import Dish


# ======================== 数据库配置 ========================
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "restaurant"),
    "charset": "utf8mb4",
}


def get_connection():
    """获取一个数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def test_connection() -> bool:
    """测试数据库连接是否正常"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False


# ======================== 菜品CRUD ========================

def fetch_all_dishes() -> list[Dish]:
    """从数据库加载所有菜品"""
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        cursor.execute("SELECT * FROM dishes ORDER BY category, id")
        rows = cursor.fetchall()
        return [_row_to_dish(row) for row in rows]
    finally:
        cursor.close()
        conn.close()


def fetch_dish_by_name(name: str) -> Optional[Dish]:
    """根据名称查找菜品（模糊匹配）"""
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        cursor.execute(
            "SELECT * FROM dishes WHERE name = %s OR name LIKE %s OR %s LIKE CONCAT('%%', name, '%%')",
            (name, f"%{name}%", name),
        )
        row = cursor.fetchone()
        if row:
            return _row_to_dish(row)
        return None
    finally:
        cursor.close()
        conn.close()


def fetch_dishes_by_category(category: str) -> list[Dish]:
    """根据分类查找菜品"""
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        cursor.execute("SELECT * FROM dishes WHERE category = %s ORDER BY id", (category,))
        rows = cursor.fetchall()
        return [_row_to_dish(row) for row in rows]
    finally:
        cursor.close()
        conn.close()


def insert_dish(dish: Dish) -> int:
    """插入一条菜品，返回影响的行数"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO dishes
                (name, price, category, spicy_level, suitable_for, dietary_tags,
                 allergens, description, is_signature, seasonal, weather_fit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            dish.name, dish.price, dish.category, dish.spicy_level,
            json.dumps(dish.suitable_for, ensure_ascii=False),
            json.dumps(dish.dietary_tags, ensure_ascii=False),
            json.dumps(dish.allergens, ensure_ascii=False),
            dish.description, dish.is_signature,
            json.dumps(dish.seasonal, ensure_ascii=False),
            json.dumps(dish.weather_fit, ensure_ascii=False),
        ))
        conn.commit()
        return cursor.rowcount
    finally:
        cursor.close()
        conn.close()


def _row_to_dish(row: dict) -> Dish:
    """将数据库行转换为Dish对象"""
    return Dish(
        id=row["id"],
        name=row["name"],
        price=float(row["price"]),
        category=row["category"],
        spicy_level=row["spicy_level"],
        suitable_for=_parse_json(row.get("suitable_for")),
        dietary_tags=_parse_json(row.get("dietary_tags")),
        allergens=_parse_json(row.get("allergens")),
        description=row["description"],
        is_signature=bool(row["is_signature"]),
        seasonal=_parse_json(row.get("seasonal")),
        weather_fit=_parse_json(row.get("weather_fit")),
    )


def _parse_json(value) -> list:
    """解析JSON字段，兼容字符串和已解析的list"""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []

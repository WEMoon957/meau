"""菜品查询模块 - 参数化查询（安全，无 LLM 生成 SQL）

原先使用 Text-to-SQL（LLM 生成自由 SQL），存在校验绕过风险：
  - 反引号表名绕过正则（``mysql`.`user``）
  - SLEEP/BENCHMARK 时间盲注
  - @@version 信息泄露
  - 子查询访问其他表

现已改为参数化查询 + 受控查询构建器，彻底消除 SQL 注入面。

对外接口保持不变：
  - query_dish_by_name(dish_name) -> str
  - list_menu_dishes(category) -> str
"""

import os
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional

from menu_data import Dish, format_dish_info
from db import get_connection, _row_to_dish


# ======================== 受控查询构建器 ========================
# 只允许在 dishes 表上按预定义字段筛选，所有值通过参数化传入，
# 不存在任何用户可控的 SQL 片段拼接。

_ALLOWED_CATEGORIES = {
    "菌彩特色", "进店必点", "经典推荐", "菌汤锅底",
    "山珍菌宴", "云岭特色", "山茅野菜",
    "普洱黄牛肉", "涮品", "甜饮品",
}
_ALLOWED_SPICY_LEVELS = {"不辣", "微辣", "中辣", "特辣"}


def _execute_query(
    sql: str,
    params: tuple,
    *,
    limit: int = 100,
    timeout_sec: float = 5.0,
) -> list[Dish]:
    """执行参数化查询（只读，带超时）

    Args:
        sql:   只含占位符 %s 的 SQL 模板（无任何用户输入拼接）
        params: 参数元组，与占位符一一对应
        limit:  最大返回行数（强制 LIMIT，防止全表扫描DoS）
        timeout_sec: 查询超时秒数

    Returns:
        Dish 对象列表
    """
    # 安全兜底：强制追加 LIMIT（如果 SQL 中还没有的话）
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {int(limit)}"

    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        # 设置查询超时（MySQL max_statement_time，5.7.4+ 支持）
        try:
            cursor.execute(
                f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_sec * 1000)}"
            )
        except pymysql.Error:
            pass  # 老版本 MySQL 不支持，忽略

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [_row_to_dish(row) for row in rows]
    finally:
        cursor.close()
        conn.close()


# ======================== 高层接口（供 tools.py 调用） ========================

def query_dish_by_name(dish_name: str) -> str:
    """按菜名查询菜品信息（参数化查询，安全）

    精确匹配优先，模糊匹配兜底，最多返回 5 条。

    Args:
        dish_name: 菜品名称

    Returns:
        格式化的菜品信息文本
    """
    if not dish_name or not dish_name.strip():
        return "请提供菜品名称。"

    name = dish_name.strip()

    # 精确匹配（参数化，完全安全）
    sql_exact = "SELECT * FROM dishes WHERE name = %s"
    dishes = _execute_query(sql_exact, (name,), limit=1)

    if dishes and len(dishes) == 1:
        return format_dish_info(dishes[0])

    # 模糊匹配（参数化 LIKE）
    if not dishes:
        sql_fuzzy = "SELECT * FROM dishes WHERE name LIKE %s"
        dishes = _execute_query(sql_fuzzy, (f"%{name}%",), limit=5)

    # 反向模糊（用户输入包含菜名）
    if not dishes:
        sql_reverse = "SELECT * FROM dishes WHERE %s LIKE CONCAT('%%', name, '%%')"
        dishes = _execute_query(sql_reverse, (name,), limit=5)

    if not dishes:
        return f"未找到菜品「{dish_name}」，请确认菜品名称。"

    if len(dishes) == 1:
        return format_dish_info(dishes[0])

    # 多条结果：简要列表
    lines = [f"找到 {len(dishes)} 道相关菜品：\n"]
    for d in dishes:
        sig = " ★" if d.is_signature else ""
        spicy = f" [{d.spicy_level}]" if d.spicy_level != "不辣" else ""
        lines.append(f"  {d.name}  ￥{d.price} ({d.category}){spicy}{sig}")
    lines.append(f"\n如需查看某道菜的详细信息，请告诉我菜品名称。")
    return "\n".join(lines)


def list_menu_dishes(category: str = "") -> str:
    """列出菜单菜品（参数化查询，安全）

    Args:
        category: 菜品分类，为空则列出全部。
                  只允许预定义分类，非法值返回提示。

    Returns:
        格式化的菜单列表文本
    """
    if category:
        category = category.strip()
        # 白名单校验：只允许预定义分类
        if category not in _ALLOWED_CATEGORIES:
            return (
                f"分类「{category}」无效，可选分类："
                f"{'、'.join(sorted(_ALLOWED_CATEGORIES))}"
            )
        sql = "SELECT * FROM dishes WHERE category = %s ORDER BY id"
        dishes = _execute_query(sql, (category,), limit=200)
        title = f"分类: {category}菜单"
    else:
        sql = "SELECT * FROM dishes ORDER BY category, id"
        dishes = _execute_query(sql, (), limit=200)
        title = "全部菜单"

    if not dishes:
        if category:
            return f"没有找到分类「{category}」的菜品，可选分类：{'、'.join(sorted(_ALLOWED_CATEGORIES))}"
        return "菜单暂无菜品。"

    lines = [f"{title}（共{len(dishes)}道）:\n"]
    current_cat = ""
    for d in dishes:
        if d.category != current_cat:
            current_cat = d.category
            lines.append(f"\n--- {current_cat} ---")
        sig = " ★" if d.is_signature else ""
        spicy = f" [{d.spicy_level}]" if d.spicy_level != "不辣" else ""
        lines.append(f"  {d.name}  ￥{d.price}{spicy}{sig}")
    return "\n".join(lines)


# ======================== 受控高级查询（替代自由 SQL） ========================

def query_dishes_advanced(
    *,
    category: str = "",
    spicy_level: str = "",
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    is_signature: Optional[bool] = None,
    suitable_for: str = "",
    dietary_tag: str = "",
    allergen: str = "",
    limit: int = 50,
) -> list[Dish]:
    """受控高级查询 - 替代原先的 LLM Text-to-SQL

    所有筛选条件通过预定义参数传入，SQL 由代码构建（非 LLM 生成），
    所有值通过 %s 参数化，不存在注入风险。

    Args:
        category:     菜品分类（白名单校验）
        spicy_level:  辣度（白名单校验）
        max_price:    最高价格
        min_price:    最低价格
        is_signature: 是否招牌菜
        suitable_for: 适合人群（JSON_CONTAINS）
        dietary_tag:  饮食标签（JSON_CONTAINS）
        allergen:     过敏原（JSON_CONTAINS）
        limit:        最大返回数

    Returns:
        Dish 对象列表
    """
    where_parts: list[str] = []
    params: list = []

    if category:
        if category not in _ALLOWED_CATEGORIES:
            return []
        where_parts.append("category = %s")
        params.append(category)

    if spicy_level:
        if spicy_level not in _ALLOWED_SPICY_LEVELS:
            return []
        where_parts.append("spicy_level = %s")
        params.append(spicy_level)

    if max_price is not None:
        where_parts.append("price <= %s")
        params.append(float(max_price))

    if min_price is not None:
        where_parts.append("price >= %s")
        params.append(float(min_price))

    if is_signature is not None:
        where_parts.append("is_signature = %s")
        params.append(bool(is_signature))

    if suitable_for:
        where_parts.append("JSON_CONTAINS(suitable_for, %s)")
        params.append(f'"{suitable_for}"')

    if dietary_tag:
        where_parts.append("JSON_CONTAINS(dietary_tags, %s)")
        params.append(f'"{dietary_tag}"')

    if allergen:
        where_parts.append("JSON_CONTAINS(allergens, %s)")
        params.append(f'"{allergen}"')

    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    sql = f"SELECT * FROM dishes WHERE {where_clause} ORDER BY id"

    return _execute_query(sql, tuple(params), limit=limit)


def format_dish_list(dishes: list[Dish], title: str = "") -> str:
    """将多个 Dish 格式化为列表展示"""
    lines = []
    if title:
        lines.append(f"{title}（共{len(dishes)}道）:\n")
    else:
        lines.append(f"共找到 {len(dishes)} 道菜品:\n")

    current_cat = ""
    for d in dishes:
        if d.category != current_cat:
            current_cat = d.category
            lines.append(f"\n--- {current_cat} ---")
        sig = " ★" if d.is_signature else ""
        spicy = f" [{d.spicy_level}]" if d.spicy_level != "不辣" else ""
        lines.append(f"  {d.name}  ￥{d.price}{spicy}{sig}")

    return "\n".join(lines)

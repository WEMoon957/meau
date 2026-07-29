"""数据库连接模块 - MySQL连接池管理与菜品数据CRUD（使用pymysql + DBUtils）

并发模型说明：
  - 全进程共享一个 PooledDB 连接池，避免每次查询新建/销毁 TCP+认证握手。
  - 池大小由 DB_POOL_SIZE 控制，超出时阻塞等待（blocking=True）以防穿透。
  - 连接在 finally 中 close() 实际是归还池，由 DBUtils 管理。
  - gunicorn 多 worker 时，每个 worker 进程独立持有一个池（fork 后不可共享连接），
    因此单 worker 池大小 × worker 数 即为 MySQL 侧总连接数，需与 MySQL
    max_connections 协调（推荐 DB_POOL_SIZE × WORKERS <= max_connections × 0.8）。
  - 所有 SELECT 注入 /*+ MAX_EXECUTION_TIME(5000) */ 优化器提示并带 LIMIT，
    防止慢查询拖垮 worker（硬约束）。
"""

import os
import json
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional

from dbutils.pooled_db import PooledDB

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

# 连接池配置
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "20"))        # 池最大连接数（单 worker）
DB_POOL_MIN_CACHED = int(os.environ.get("DB_POOL_MIN_CACHED", "2"))
DB_POOL_MAX_CACHED = int(os.environ.get("DB_POOL_MAX_CACHED", "5"))
DB_CONN_RECYCLE = int(os.environ.get("DB_CONN_RECYCLE", "1800"))  # 连接最大存活秒数（防 MySQL 8h 断连）

# 查询超时（毫秒）——与项目硬约束一致
DB_MAX_EXECUTION_MS = int(os.environ.get("DB_MAX_EXECUTION_MS", "5000"))

# 全表扫描兜底行数
DISH_LIMIT = int(os.environ.get("DB_DISH_LIMIT", "1000"))

_pool: Optional[PooledDB] = None


def _get_pool() -> PooledDB:
    """懒加载连接池。

    懒加载原因：gunicorn preload_app=True 时模块在 master 进程导入，
    若此时建池会被 fork 的子进程共享连接（MySQL 协议不允许），故推迟到首次使用。
    """
    global _pool
    if _pool is None:
        _pool = PooledDB(
            creator=pymysql,
            maxconnections=DB_POOL_SIZE,
            mincached=DB_POOL_MIN_CACHED,
            maxcached=DB_POOL_MAX_CACHED,
            maxshared=0,            # 不在多线程间共享同一连接（pymysql 非线程安全）
            blocking=True,          # 池耗尽时阻塞等待，而非直接报错（背压）
            ping=1,                 # 取连接时校验存活，自动剔除死连接
            maxusage=DB_CONN_RECYCLE,
            **DB_CONFIG,
        )
    return _pool


def get_connection():
    """从池中获取一个连接（用完必须 close() 归还池）。"""
    return _get_pool().connection()


def close_pool() -> None:
    """关闭并释放连接池（仅在进程退出时调用）。"""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None


def test_connection() -> bool:
    """测试数据库连接是否正常（使用池）。"""
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
# 注意：所有 SELECT 均带 MAX_EXECUTION_TIME 优化器提示 + LIMIT（硬约束）

def fetch_all_dishes() -> list[Dish]:
    """从数据库加载所有菜品"""
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        cursor.execute(
            f"SELECT /*+ MAX_EXECUTION_TIME({DB_MAX_EXECUTION_MS}) */ * "
            f"FROM dishes ORDER BY category, id LIMIT %s",
            (DISH_LIMIT,),
        )
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
            f"SELECT /*+ MAX_EXECUTION_TIME({DB_MAX_EXECUTION_MS}) */ * "
            f"FROM dishes WHERE name = %s OR name LIKE %s OR %s LIKE CONCAT('%%', name, '%%') "
            f"LIMIT %s",
            (name, f"%{name}%", name, 5),
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
        cursor.execute(
            f"SELECT /*+ MAX_EXECUTION_TIME({DB_MAX_EXECUTION_MS}) */ * "
            f"FROM dishes WHERE category = %s ORDER BY id LIMIT %s",
            (category, DISH_LIMIT),
        )
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
        price=float(row["price"]) if row["price"] is not None else 0.0,
        category=row["category"],
        spicy_level=row["spicy_level"],
        suitable_for=_parse_json(row.get("suitable_for")),
        dietary_tags=_parse_json(row.get("dietary_tags")),
        allergens=_parse_json(row.get("allergens")),
        description=row["description"],
        is_signature=bool(row["is_signature"]),
        seasonal=_parse_json(row.get("seasonal")),
        weather_fit=_parse_json(row.get("weather_fit")),
        gross_margin=float(row["gross_margin"]) if row.get("gross_margin") is not None else 0.0,
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

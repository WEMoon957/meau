"""Text-to-SQL 模块 - 使用 LLM 将自然语言转换为 SQL 查询菜品数据

将自然语言问题转换为 SQL SELECT 查询，在 MySQL 数据库上执行，
返回格式化的菜品信息。仅允许只读查询（SELECT），禁止任何写操作。

流程：自然语言 → LLM 生成 SQL → 安全校验 → 执行查询 → 格式化结果
"""

import os
import re
import pymysql
from pymysql.cursors import DictCursor
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from menu_data import Dish, format_dish_info
from db import get_connection, _row_to_dish


# ======================== 数据库 Schema 描述（供 LLM 生成 SQL） ========================
SCHEMA_DESCRIPTION = """数据库: restaurant
表: dishes (菜品表)

| 列名 | 类型 | 说明 |
|------|------|------|
| id | INT | 主键，自增 |
| name | VARCHAR(100) | 菜品名称 |
| price | DECIMAL(10,2) | 价格（元） |
| category | VARCHAR(20) | 分类，可选值: 凉菜/热菜/汤品/主食/饮品/甜点 |
| spicy_level | VARCHAR(20) | 辣度，可选值: 不辣/微辣/中辣/特辣 |
| suitable_for | JSON | 适合人群列表，如 ["聚餐","儿童","老人","情侣","一人食","下酒"] |
| dietary_tags | JSON | 饮食标签列表，如 ["素食","低脂","低糖","高蛋白","无麸质"] |
| allergens | JSON | 过敏原列表，如 ["花生","海鲜","鸡蛋","牛奶","大豆"] |
| description | TEXT | 菜品介绍 |
| is_signature | BOOLEAN | 是否招牌菜（TRUE/FALSE） |
| seasonal | JSON | 适合季节列表，如 ["春","夏","秋","冬"] |
| weather_fit | JSON | 适合天气列表，如 ["热天","冷天","雨天"] |

JSON 列查询方式（MySQL）:
- 判断 JSON 数组是否包含某值: JSON_CONTAINS(suitable_for, '"儿童"')
- 模糊搜索 JSON 数组内容: JSON_EXTRACT(suitable_for, '$') LIKE '%儿童%'
"""


SQL_SYSTEM_PROMPT = f"""你是一个 SQL 生成专家。根据用户的自然语言问题，生成对应的 MySQL SELECT 查询语句。

{SCHEMA_DESCRIPTION}

规则:
1. 只能生成 SELECT 语句，绝对禁止 INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE 等写操作
2. 使用 MySQL 语法
3. 查询 JSON 数组是否包含某值时，使用 JSON_CONTAINS(列名, '"值"')
4. 模糊匹配菜品名称时，使用 LIKE '%关键词%'
5. 只输出一条 SQL 语句，不要加任何解释，不要用 markdown 代码块包裹
6. 查询所有列时使用 SELECT *
7. 默认按 id 排序，除非用户明确要求其他排序方式
8. 不要以分号结尾

示例:
用户: 查找名为宫保鸡丁的菜品
SQL: SELECT * FROM dishes WHERE name = '宫保鸡丁'

用户: 查找所有热菜
SQL: SELECT * FROM dishes WHERE category = '热菜' ORDER BY id

用户: 价格低于30的不辣菜品
SQL: SELECT * FROM dishes WHERE price < 30 AND spicy_level = '不辣' ORDER BY id

用户: 适合儿童的招牌菜
SQL: SELECT * FROM dishes WHERE is_signature = TRUE AND JSON_CONTAINS(suitable_for, '"儿童"') ORDER BY id

用户: 所有菜品按分类排序
SQL: SELECT * FROM dishes ORDER BY category, id

用户: 查找名字包含鱼的菜
SQL: SELECT * FROM dishes WHERE name LIKE '%鱼%' ORDER BY id

用户: 查找名为宫保鸡丁的菜品，精确匹配不到时模糊匹配
SQL: SELECT * FROM dishes WHERE name = '宫保鸡丁' OR name LIKE '%宫保鸡丁%' OR '宫保鸡丁' LIKE CONCAT('%', name, '%') LIMIT 5
"""

# 禁止的 SQL 关键词（不区分大小写）
# 含 UNION/INTO/DUMPFILE：UNION 可跨表读取，INTO OUTFILE/DUMPFILE 可写文件
_FORBIDDEN_KEYWORDS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|EXEC|MERGE|REPLACE|CALL|LOAD|OUTFILE|DUMPFILE|UNION|INTO)\b',
    re.IGNORECASE,
)


# ======================== LLM 初始化 ========================
_llm_instance: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    """懒加载 LLM 实例（从环境变量读取配置）"""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "qwen-turbo")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    kwargs = {"model": model, "api_key": api_key, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url

    _llm_instance = ChatOpenAI(**kwargs)
    return _llm_instance


# ======================== SQL 生成 ========================
def generate_sql(question: str) -> str:
    """使用 LLM 将自然语言问题转换为 SQL 语句

    Args:
        question: 自然语言查询问题

    Returns:
        生成的 SQL 字符串
    """
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=SQL_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])

    sql = response.content.strip()
    # 去除可能的 markdown 代码块标记
    sql = re.sub(r'^```(?:sql)?\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    return sql.strip().rstrip(";")


# ======================== SQL 安全校验 ========================
def validate_sql(sql: str) -> tuple[bool, str]:
    """验证 SQL 语句是否安全（仅允许只读 SELECT）

    Args:
        sql: 待校验的 SQL 语句

    Returns:
        (is_valid, error_message)
    """
    if not sql or not sql.strip():
        return False, "SQL 语句为空"

    sql_stripped = sql.strip()
    sql_upper = sql_stripped.upper()

    # 必须以 SELECT 开头
    if not sql_upper.startswith("SELECT"):
        return False, "仅允许 SELECT 查询"

    # 检查禁止的关键词
    match = _FORBIDDEN_KEYWORDS.search(sql_stripped)
    if match:
        return False, f"SQL 包含禁止的关键词: {match.group()}"

    # 不允许分号（防止多语句注入）
    if ";" in sql_stripped:
        return False, "SQL 不允许包含分号"

    # 不允许注释
    if "--" in sql_stripped or "/*" in sql_stripped or "#" in sql_stripped:
        return False, "SQL 不允许包含注释"

    # 仅允许查询 dishes 表，禁止通过子查询/JOIN/UNION 读取其他表
    # （UNION 已被关键词拦截，此处兜底拦截 FROM/JOIN 引用的其他表）
    referenced_tables = re.findall(r'\bFROM\s+([\w.]+)', sql_stripped, re.IGNORECASE)
    referenced_tables += re.findall(r'\bJOIN\s+([\w.]+)', sql_stripped, re.IGNORECASE)
    for t in referenced_tables:
        if t.split('.')[-1].lower() != 'dishes':
            return False, f"仅允许查询 dishes 表，禁止访问: {t}"

    return True, ""


# ======================== SQL 执行 ========================
def execute_sql(sql: str) -> list[Dish]:
    """执行 SQL 查询并返回 Dish 对象列表

    Args:
        sql: 已校验的 SELECT SQL 语句

    Returns:
        查询到的 Dish 列表
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        cursor.execute(sql)
        rows = cursor.fetchall()
        return [_row_to_dish(row) for row in rows]
    finally:
        cursor.close()
        conn.close()


# ======================== 结果格式化 ========================
def _format_dish_list(dishes: list[Dish], title: str = "") -> str:
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


# ======================== 高层接口（供 tools.py 调用） ========================
def query_dish_by_name(dish_name: str) -> str:
    """通过 text-to-sql 查询菜品信息（供 query_dish 工具调用）

    生成 SQL 查找匹配菜品名称的记录，精确匹配优先，模糊匹配兜底。

    Args:
        dish_name: 菜品名称

    Returns:
        格式化的菜品信息文本
    """
    question = f"查找名为{dish_name}的菜品，精确匹配不到时用名称模糊匹配，最多返回5条"

    # 1. 生成 SQL
    try:
        sql = generate_sql(question)
    except Exception as e:
        return f"SQL 生成失败: {e}"

    # 2. 校验 SQL
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return f"SQL 校验失败: {error}"

    # 3. 执行查询
    try:
        dishes = execute_sql(sql)
    except Exception as e:
        return f"数据库查询失败: {e}"

    # 4. 格式化结果
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
    """通过 text-to-sql 列出菜单菜品（供 list_menu 工具调用）

    Args:
        category: 菜品分类，为空则列出全部

    Returns:
        格式化的菜单列表文本
    """
    if category:
        question = f"查找所有分类为{category}的菜品，按id排序"
        title = f"分类: {category}菜单"
    else:
        question = "查找所有菜品，按分类和id排序"
        title = "全部菜单"

    # 1. 生成 SQL
    try:
        sql = generate_sql(question)
    except Exception as e:
        return f"SQL 生成失败: {e}"

    # 2. 校验 SQL
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return f"SQL 校验失败: {error}"

    # 3. 执行查询
    try:
        dishes = execute_sql(sql)
    except Exception as e:
        return f"数据库查询失败: {e}"

    # 4. 格式化结果
    if not dishes:
        if category:
            return f"没有找到分类「{category}」的菜品，可选分类：凉菜/热菜/汤品/主食/饮品/甜点"
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

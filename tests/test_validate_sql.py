"""validate_sql 安全校验单元测试

覆盖 text_to_sql.validate_sql：
- 合法 SELECT 查询应通过
- 跨表读取 / UNION / 文件写入 / 多语句 / 注释 等攻击向量应被拦截

注：text_to_sql 顶层导入了 pymysql / langchain 等重量级依赖，
本测试在导入前用 sys.modules 注入轻量桩，避免对 DB/网络/LLM 的依赖，
从而可以直接测试 validate_sql 的纯函数逻辑。
"""
import sys
import types
import os

# ---- 注入轻量桩，绕过重量级依赖 ----
# pymysql
_pymysql = types.ModuleType("pymysql")
_pymysql.connect = lambda **k: None
sys.modules.setdefault("pymysql", _pymysql)
_pymysql_cursors = types.ModuleType("pymysql.cursors")
_pymysql_cursors.DictCursor = object
sys.modules.setdefault("pymysql.cursors", _pymysql_cursors)
# langchain_openai
sys.modules.setdefault("langchain_openai", types.ModuleType("langchain_openai"))
sys.modules["langchain_openai"].ChatOpenAI = type("ChatOpenAI", (), {})
# langchain_core / langchain_core.messages
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
_lc_messages = types.ModuleType("langchain_core.messages")
_lc_messages.HumanMessage = type("HumanMessage", (), {})
_lc_messages.SystemMessage = type("SystemMessage", (), {})
sys.modules.setdefault("langchain_core.messages", _lc_messages)

# 将 main 目录加入路径后导入被测模块
_MAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main")
sys.path.insert(0, _MAIN_DIR)

from text_to_sql import validate_sql  # noqa: E402


# ---- 合法查询：必须通过 ----
LEGIT_QUERIES = [
    "SELECT * FROM dishes WHERE name = '宫保鸡丁'",
    "SELECT * FROM dishes WHERE category = '热菜' ORDER BY id",
    "SELECT * FROM dishes WHERE price < 30 AND spicy_level = '不辣' ORDER BY id",
    "SELECT * FROM dishes WHERE is_signature = TRUE AND JSON_CONTAINS(suitable_for, '\"儿童\"') ORDER BY id",
    "SELECT * FROM dishes ORDER BY category, id",
    "SELECT * FROM dishes WHERE name LIKE '%鱼%' ORDER BY id",
    "SELECT * FROM dishes WHERE name = '宫保鸡丁' OR name LIKE '%宫保鸡丁%' OR '宫保鸡丁' LIKE CONCAT('%', name, '%') LIMIT 5",
    "SELECT * FROM restaurant.dishes ORDER BY id",  # 带库名限定
]

# ---- 攻击向量：必须被拦截 ----
ATTACK_QUERIES = [
    # UNION 跨表读取（如读取 mysql 用户口令哈希）
    "SELECT * FROM dishes WHERE name = 'x' UNION SELECT user,authentication_string,1,1,1,1,1,1,1,1,1,1 FROM mysql.user",
    # INTO OUTFILE / DUMPFILE 写文件
    "SELECT * FROM dishes INTO OUTFILE '/tmp/x'",
    "SELECT * FROM dishes INTO DUMPFILE '/tmp/x'",
    # 跨表子查询
    "SELECT * FROM dishes WHERE id IN (SELECT id FROM mysql.user)",
    # 直接查询其他表
    "SELECT * FROM information_schema.tables",
    # 跨表 JOIN
    "SELECT * FROM dishes JOIN mysql.user ON 1=1",
    # 多语句
    "SELECT * FROM dishes; DROP TABLE dishes",
    # 注释
    "SELECT * FROM dishes-- comment",
    # 非 SELECT 写操作
    "DELETE FROM dishes",
]


def test_legit_queries_pass():
    for q in LEGIT_QUERIES:
        valid, err = validate_sql(q)
        assert valid, f"合法查询被误拦: {q}\n  原因: {err}"


def test_attack_queries_blocked():
    for q in ATTACK_QUERIES:
        valid, err = validate_sql(q)
        assert not valid, f"攻击向量未被拦截: {q}"


def test_empty_rejected():
    assert validate_sql("")[0] is False
    assert validate_sql("   ")[0] is False


if __name__ == "__main__":
    # 简单自跑，便于在无 pytest 的环境验证
    test_legit_queries_pass()
    test_attack_queries_blocked()
    test_empty_rejected()
    print("ALL TESTS PASSED")

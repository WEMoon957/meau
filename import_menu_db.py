"""
菌彩菜品作业指导书 - 结构化数据库导入脚本（MySQL版）

从Excel文件解析菜品的SOP数据（菜名/售价/毛利/成本/四步制作流程），
存入MySQL的 dishes_sop 表（先删旧表再建新表，保证幂等）。

数据库表结构：
  dishes_sop - 菜品SOP主表（菜名/售价/毛利/成本/选料要求/刀功成型/制作流程/技术关键）

使用方式：
  python import_menu_db.py [excel文件路径]

  不传参数时使用下方默认路径。可通过命令行指定其他Excel文件。

环境变量配置（从 .env 读取，与 main/db.py 一致）：
  DB_HOST     MySQL主机地址（默认 localhost）
  DB_PORT     MySQL端口（默认 3306）
  DB_USER     用户名（默认 root）
  DB_PASSWORD 密码（默认空）
  DB_NAME     数据库名（默认 restaurant）
"""

import os
import sys
import pymysql
import pandas as pd

# ============================================================
# 加载 .env 文件（与 config.py 一致的手动解析，不依赖 python-dotenv）
# ============================================================
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ============================================================
# Excel文件路径（支持命令行参数）
# ============================================================
DEFAULT_EXCEL_PATH = os.path.join(
    r"c:\Users\work\.trae-cn\attachments\6a683aaebf3e3e8c2cdee22d",
    "24cff62c-3131-4ff8-abd7-6efe035e88b7_305950ef-b985-42d1-9404-f7f57452e194_菜品作业指导书-2.0版xlsx.xlsx北京区域.xlsx",
)
EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXCEL_PATH

# ============================================================
# 数据库配置（与 main/db.py 保持一致，从环境变量读取）
# ============================================================
DB_SERVER_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "charset": "utf8mb4",
}
DB_NAME = os.environ.get("DB_NAME", "restaurant")

# 需要跳过的非菜品Sheet
SKIP_SHEETS = {
    "表一产品SOP", "Sheet3", "WpsReserved_CellImgList", "表二原材料定价",
    "表一产品SOP (2)", "表一产品SOP (3)", "表一产品SOP (4)", "表一产品SOP (5)",
    "表一产品SOP (6)", "表一产品SOP (7)", "表一产品SOP (8)", "表一产品SOP (9)",
    "表一产品SOP (10)", "表一产品SOP (11)", "表一产品SOP (12)", "表一产品SOP (13)",
    "表一产品SOP (14)", "表一产品SOP (15)", "表一产品SOP (16)",
}


def _safe_str(val):
    """安全转换为字符串"""
    if pd.isna(val):
        return ""
    return str(val).strip()


def _safe_float(val):
    """安全转换为浮点数"""
    if pd.isna(val) or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ============================================================
# 数据库初始化
# ============================================================
def create_database():
    """创建数据库（如不存在）"""
    conn = pymysql.connect(**DB_SERVER_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cursor.close()
    conn.close()
    print(f"  数据库 '{DB_NAME}' 已创建/确认存在")


def init_tables():
    """创建 dishes_sop 表（先删旧表再建新表，保证幂等）"""
    conn = pymysql.connect(database=DB_NAME, **DB_SERVER_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS dishes_sop")

    # 菜品SOP主表
    cursor.execute("""
        CREATE TABLE dishes_sop (
            id                    INT AUTO_INCREMENT PRIMARY KEY,
            name                  VARCHAR(200)  NOT NULL COMMENT '菜名',
            price                 DECIMAL(10,2) COMMENT '售价（元）',
            gross_margin          DECIMAL(5,2)  COMMENT '毛利率',
            total_cost            DECIMAL(10,2) COMMENT '成本合计（元）',
            selection_requirement TEXT          COMMENT '第一步：选料要求',
            cutting_shape         TEXT          COMMENT '第二步：刀功成型',
            cooking_process       TEXT          COMMENT '第三步：制作流程',
            technical_key         TEXT          COMMENT '第四步：技术关键',
            created_at            TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_dish_name (name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='菜品SOP主表'
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("  表结构已创建（dishes_sop）")


# ============================================================
# 解析菜品SOP Sheet
# ============================================================
def parse_dish_sop(xls, sheet_name):
    """
    解析单个菜品SOP Sheet

    返回: dish_info 或 None
    """
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    # Row 1: [0]=菜名 [1]=菜名值 [3]=售价 [5]=毛利
    dish_name = _safe_str(df.iloc[1, 1]) if df.shape[1] > 1 else ""
    if not dish_name:
        dish_name = sheet_name.strip()

    price = _safe_float(df.iloc[1, 3]) if df.shape[1] > 3 else None
    gross_margin = _safe_float(df.iloc[1, 5]) if df.shape[1] > 5 else None

    # 成本合计在 Row 2, col 13
    total_cost = _safe_float(df.iloc[2, 13]) if df.shape[1] > 13 else None

    # 制作步骤
    selection_requirement = ""
    cutting_shape = ""
    cooking_process = ""
    technical_key = ""

    for idx in range(len(df)):
        col0 = _safe_str(df.iloc[idx, 0]) if df.shape[1] > 0 else ""
        col2 = _safe_str(df.iloc[idx, 2]) if df.shape[1] > 2 else ""

        if "第一步" in col0 and "选料" in col0:
            selection_requirement = col2
        elif "第二步" in col0 and "刀功" in col0:
            cutting_shape = col2
        elif "第三" in col0 and "制作" in col0:
            cooking_process = col2
        elif "第四" in col0 and "技术" in col0:
            technical_key = col2

    dish_info = {
        "name": dish_name,
        "price": price,
        "gross_margin": gross_margin,
        "total_cost": total_cost,
        "selection_requirement": selection_requirement,
        "cutting_shape": cutting_shape,
        "cooking_process": cooking_process,
        "technical_key": technical_key,
    }

    return dish_info


# ============================================================
# 主导入流程
# ============================================================
def main():
    print("=" * 60)
    print("菌彩菜品作业指导书 - 结构化数据库导入（MySQL）")
    print("=" * 60)
    print(f"  Excel文件: {os.path.basename(EXCEL_PATH)}")
    print(f"  数据库: {DB_SERVER_CONFIG['host']}:{DB_SERVER_CONFIG['port']}/{DB_NAME}")
    print(f"  用户: {DB_SERVER_CONFIG['user']}")
    print("=" * 60)

    # 1. 读取Excel
    print("\n[1/4] 读取Excel文件...")
    if not os.path.exists(EXCEL_PATH):
        print(f"  ❌ Excel文件不存在: {EXCEL_PATH}")
        sys.exit(1)
    xls = pd.ExcelFile(EXCEL_PATH)
    print(f"  共 {len(xls.sheet_names)} 个Sheet")

    # 2. 创建数据库
    print("\n[2/4] 创建数据库...")
    create_database()

    # 3. 创建表结构
    print("\n[3/4] 初始化表结构...")
    init_tables()

    # 4. 导入菜品SOP数据
    print("\n[4/4] 导入菜品SOP数据...")
    conn = pymysql.connect(database=DB_NAME, **DB_SERVER_CONFIG)
    cursor = conn.cursor()

    dish_count = 0
    skip_count = 0

    dish_sheets = [s for s in xls.sheet_names if s not in SKIP_SHEETS]
    print(f"  菜品Sheet数量: {len(dish_sheets)}")

    # REPLACE INTO：遇到同名菜品先删后插，保证用最新数据覆盖
    insert_dish_sql = """
        REPLACE INTO dishes_sop
            (name, price, gross_margin, total_cost,
             selection_requirement, cutting_shape,
             cooking_process, technical_key)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    for sheet_name in dish_sheets:
        try:
            dish_info = parse_dish_sop(xls, sheet_name)
            if dish_info is None:
                skip_count += 1
                continue
            if not dish_info["name"]:
                skip_count += 1
                continue

            cursor.execute(insert_dish_sql, (
                dish_info["name"],
                dish_info["price"],
                dish_info["gross_margin"],
                dish_info["total_cost"],
                dish_info["selection_requirement"],
                dish_info["cutting_shape"],
                dish_info["cooking_process"],
                dish_info["technical_key"],
            ))
            conn.commit()  # 逐条手动提交
            dish_count += 1

        except Exception as e:
            print(f"  ⚠️ Sheet '{sheet_name}' 导入失败: {e}")
            conn.rollback()
            skip_count += 1

    print(f"  成功导入: {dish_count} 道  跳过: {skip_count} 个")

    # ============================================================
    # 数据验证
    # ============================================================
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)

    cursor.execute("SELECT COUNT(*) FROM dishes_sop")
    print(f"菜品SOP: {cursor.fetchone()[0]} 条")

    # 抽样验证
    print(f"\n抽样验证 - 菜品前5条:")
    cursor.execute("""
        SELECT name, price, gross_margin, total_cost
        FROM dishes_sop
        ORDER BY id
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: 售价={row[1]} 毛利={row[2]} 成本={row[3]}")

    cursor.close()
    conn.close()
    print(f"\n导入完成！数据库: {DB_NAME}")


if __name__ == "__main__":
    main()

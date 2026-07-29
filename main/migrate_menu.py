"""
菌彩菜单迁移脚本 - 从 ChromaDB 知识库 + dishes_sop 合并写入 dishes 表

背景：
  Agent 的 query_dish、list_menu 等工具依赖 MySQL dishes 表，
  但该表从未被真实数据填充，导致推荐回退到 27 道硬编码的通用菜品。

步骤：
  1. 从 ChromaDB 读取 81 道菜品的档案（辣度/适合人群/过敏原等）
  2. 从 dishes_sop 读取 90 道菜品的定价
  3. 按菜名合并，写入 dishes 表
"""

import json
import os
import re
import sys
import pymysql

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)  # 进入 main/
sys.path.insert(0, _PROJECT_ROOT) # 项目根目录（可导入 kb_store 等）

# 手动加载 .env
_env_path = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "charset": "utf8mb4",
    "autocommit": False,  # 手动控制事务提交
}
DB_NAME = os.environ.get("DB_NAME", "restaurant")


def get_db():
    return pymysql.connect(database=DB_NAME, **DB_CONFIG)


# ------------------------------------------------------------------
# 1. 从 ChromaDB 加载菜品档案
# ------------------------------------------------------------------
def load_chroma_dishes() -> dict[str, dict]:
    """返回 {菜品名: {...}}"""
    from kb_store import VectorStore

    store = VectorStore()
    try:
        store.load()
    except FileNotFoundError:
        print("❌ ChromaDB 知识库未构建，请先运行 python build_kb.py")
        sys.exit(1)

    out = {}
    for m in store._metadatas_cache:
        if m.get("type") != "dish_profile":
            continue
        name = m.get("dish_name", "").strip()
        if not name:
            continue

        # 解析适合人群："老人、小孩" → ["老人", "小孩"]
        crowd_raw = m.get("suitable_crowd", "")
        suitable_for = [c.strip() for c in re.split(r"[、，,]", crowd_raw) if c.strip()]

        # 解析过敏原：只取"含X"（不含的忽略）
        allergen_raw = m.get("allergen_info", "")
        allergens = []
        for part in re.split(r"[，,]", allergen_raw):
            part = part.strip()
            if part.startswith("含") and not part.startswith("不含"):
                allergens.append(part[1:])

        # 从 text 中提取 description（metadata 里没有独立描述字段）
        # 用 text 字段 + metadata 拼接描述
        desc_parts = []
        cal = m.get("calorie", "")
        prop = m.get("property", "")
        pairing = m.get("pairing", "")
        if prop:
            desc_parts.append(prop)
        if cal:
            desc_parts.append(cal)
        if pairing:
            desc_parts.append(f"搭配建议：{pairing}")

        # 饮食标签
        dietary_tags = []
        if "极低热量" in cal:
            dietary_tags.append("低热量")
        if "低热量" in cal and "极低" not in cal:
            dietary_tags.append("低热量")
        if "素" in prop or "素菜" in prop:
            dietary_tags.append("素食")
        if prop:
            dietary_tags.append(prop)

        out[name] = {
            "name": name,
            "category": m.get("category", ""),
            "spicy_level": m.get("spice_level", "不辣"),
            "suitable_for": suitable_for,
            "allergens": allergens,
            "dietary_tags": dietary_tags,
            "description": "；".join(desc_parts),
        }
    return out


# ------------------------------------------------------------------
# 2. 从 dishes_sop 加载定价
# ------------------------------------------------------------------
def load_sop_prices() -> dict[str, dict]:
    """返回 {菜品名: {price, gross_margin, total_cost}}"""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT name, price, gross_margin, total_cost FROM dishes_sop")
    out = {}
    for row in c.fetchall():
        name = row[0].strip() if row[0] else ""
        if not name:
            continue
        # gross_margin 是 DECIMAL(5,2)，范围 0.62~1.00；NULL 归一化为 0
        gm = float(row[2]) if row[2] is not None else 0.0
        out[name] = {
            "price": row[1],
            "gross_margin": gm,
            "total_cost": row[3],
        }
    c.close()
    conn.close()
    return out


# ------------------------------------------------------------------
# 3. 创建 dishes 表（先删旧表再建新表）
# ------------------------------------------------------------------
def ensure_dishes_table(conn):
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS dishes")
    c.execute("""
        CREATE TABLE dishes (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            name          VARCHAR(100)  NOT NULL COMMENT '菜品名称',
            price         DECIMAL(10,2) COMMENT '价格',
            gross_margin  DECIMAL(5,2)  COMMENT '毛利率(0~1)',
            category      VARCHAR(50)   NOT NULL DEFAULT '' COMMENT '分类',
            spicy_level   VARCHAR(20)   NOT NULL DEFAULT '不辣' COMMENT '辣度',
            suitable_for  JSON          COMMENT '适合人群列表',
            dietary_tags  JSON          COMMENT '饮食标签',
            allergens     JSON          COMMENT '过敏原列表',
            description   TEXT          COMMENT '菜品介绍',
            is_signature  BOOLEAN       NOT NULL DEFAULT FALSE,
            seasonal      JSON          COMMENT '适合季节',
            weather_fit   JSON          COMMENT '适合天气',
            created_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_name (name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='菜品表'
    """)
    conn.commit()
    c.close()
    print("✅ dishes 表已重建（含 gross_margin 字段）")


# ------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("菌彩真实菜单 → dishes 表 迁移")
    print("=" * 60)

    # 加载数据源
    print("\n[1/3] 从 ChromaDB 加载菜品档案...")
    chroma_dishes = load_chroma_dishes()
    print(f"  知识库菜品: {len(chroma_dishes)} 道")

    print("\n[2/3] 从 dishes_sop 加载定价...")
    sop_prices = load_sop_prices()
    print(f"  SOP 菜品: {len(sop_prices)} 道")

    # 合并
    print("\n[3/3] 合并写入 dishes 表...")
    conn = get_db()
    ensure_dishes_table(conn)
    cur = conn.cursor()

    all_names = set(chroma_dishes.keys()) | set(sop_prices.keys())
    insert_count = 0
    price_missing = 0  # 在 chroma 中有但在 dishes_sop 中无价格
    name_only = 0  # 仅在 dishes_sop 中有

    for name in sorted(all_names):
        kb = chroma_dishes.get(name, {})
        sop = sop_prices.get(name, {})

        dish_name = name
        price = sop.get("price")
        gross_margin = sop.get("gross_margin", 0.0)
        category = kb.get("category", "")
        spicy_level = kb.get("spicy_level", "不辣")
        suitable_for = kb.get("suitable_for", [])
        dietary_tags = kb.get("dietary_tags", [])
        allergens = kb.get("allergens", [])
        description = kb.get("description", "")

        # 补齐 category（仅有价格的数据）
        if not category:
            category = "菌彩特色"

        if not kb:
            name_only += 1

        if not sop:
            price_missing += 1

        cur.execute(
            """INSERT INTO dishes
               (name, price, gross_margin, category, spicy_level, suitable_for,
                dietary_tags, allergens, description)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                dish_name,
                price,
                gross_margin,
                category,
                spicy_level,
                json.dumps(suitable_for, ensure_ascii=False),
                json.dumps(dietary_tags, ensure_ascii=False),
                json.dumps(allergens, ensure_ascii=False),
                description,
            ),
        )
        insert_count += 1

    conn.commit()

    print(f"\n  已写入 dishes 表: {insert_count} 道菜品")
    if price_missing:
        print(f"  ⚠️ 知识库有但缺定价: {price_missing} 道（需在 dishes_sop 补充）")
    if name_only:
        print(f"  ℹ️ 仅有定价无属性: {name_only} 道（不在知识库中）")

    # 验证
    cur.execute("SELECT COUNT(*) FROM dishes")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM dishes WHERE price IS NOT NULL")
    priced = cur.fetchone()[0]

    print(f"\n{'=' * 60}")
    print(f"dishes 表总计: {total} 道（其中 {priced} 道有定价）")
    print(f"{'=' * 60}")

    # 展示前 15 条
    cur.execute("SELECT name, price, category, spicy_level FROM dishes ORDER BY id LIMIT 15")
    print("\n前 15 道菜品：")
    for row in cur.fetchall():
        price_str = f"¥{row[1]}" if row[1] else "未定价"
        print(f"  {row[0]:20s}  {price_str:>8s}  {row[2]:15s}  {row[3]}")

    cur.close()
    conn.close()
    print(f"\n✅ 迁移完成！请重启 API 服务器使新数据生效。")


if __name__ == "__main__":
    main()

"""数据库初始化脚本 - 建库、建表、导入初始菜品数据

使用方式：
    python main/init_db.py
"""

import os
import sys
import json
import pymysql

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from menu_data import MENU, invalidate_dishes_cache

# 数据库连接配置（不指定database，用于创建数据库）
DB_SERVER_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "charset": "utf8mb4",
}
DB_NAME = os.environ.get("DB_NAME", "restaurant")


def create_database():
    """创建数据库"""
    conn = pymysql.connect(**DB_SERVER_CONFIG)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.close()
    conn.close()
    print(f"✅ 数据库 '{DB_NAME}' 已创建/确认存在")


def create_tables():
    """创建菜品表"""
    conn = pymysql.connect(database=DB_NAME, **DB_SERVER_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dishes (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            name          VARCHAR(100)  NOT NULL COMMENT '菜品名称',
            price         DECIMAL(10,2) NOT NULL COMMENT '价格',
            category      VARCHAR(20)   NOT NULL COMMENT '分类: 凉菜/热菜/汤品/主食/饮品/甜点',
            spicy_level   VARCHAR(20)   NOT NULL DEFAULT '不辣' COMMENT '辣度: 不辣/微辣/中辣/特辣',
            suitable_for  JSON          COMMENT '适合人群列表',
            dietary_tags  JSON          COMMENT '饮食标签列表: 素食/低脂/低糖/高蛋白/无麸质',
            allergens     JSON          COMMENT '过敏原列表: 花生/海鲜/鸡蛋/牛奶/大豆',
            description   TEXT          COMMENT '菜品介绍',
            is_signature  BOOLEAN       NOT NULL DEFAULT FALSE COMMENT '是否招牌菜',
            seasonal      JSON          COMMENT '适合季节: 春/夏/秋/冬',
            weather_fit   JSON          COMMENT '适合天气: 热天/冷天/雨天',
            created_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_name (name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='菜品表'
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 表 'dishes' 已创建/确认存在")


def import_data():
    """导入初始菜品数据"""
    conn = pymysql.connect(database=DB_NAME, **DB_SERVER_CONFIG)
    cursor = conn.cursor()

    # 先清空旧数据
    cursor.execute("DELETE FROM dishes")
    conn.commit()

    sql = """
        INSERT INTO dishes
            (id, name, price, category, spicy_level, suitable_for, dietary_tags,
             allergens, description, is_signature, seasonal, weather_fit)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for dish in MENU:
        cursor.execute(sql, (
            dish.id, dish.name, dish.price, dish.category, dish.spicy_level,
            json.dumps(dish.suitable_for, ensure_ascii=False),
            json.dumps(dish.dietary_tags, ensure_ascii=False),
            json.dumps(dish.allergens, ensure_ascii=False),
            dish.description, dish.is_signature,
            json.dumps(dish.seasonal, ensure_ascii=False),
            json.dumps(dish.weather_fit, ensure_ascii=False),
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ 已导入 {len(MENU)} 道菜品数据")
    invalidate_dishes_cache()


def main():
    print("=" * 50)
    print("  餐厅数据库初始化脚本")
    print("=" * 50)
    print(f"  主机: {DB_SERVER_CONFIG['host']}:{DB_SERVER_CONFIG['port']}")
    print(f"  用户: {DB_SERVER_CONFIG['user']}")
    print(f"  数据库: {DB_NAME}")
    print("=" * 50)

    try:
        create_database()
        create_tables()
        import_data()
        print("\n🎉 数据库初始化完成！")
    except pymysql.Error as e:
        print(f"\n❌ 数据库错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

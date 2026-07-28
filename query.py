"""
菌彩野生菌火锅 · 菜品推荐知识库 - 查询接口

使用方式：
  1. 交互模式：python query.py
  2. 直接查询：python query.py "适合老人吃不辣的菜"
  3. 代码调用：from query import KnowledgeBaseQuery
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    API_KEY, BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    KB_FILE, CHROMA_DIR,
)
from kb_store import VectorStore
from openai import OpenAI


class KnowledgeBaseQuery:
    """菜品推荐知识库查询器"""

    def __init__(self):
        """加载知识库"""
        if not os.path.exists(CHROMA_DIR):
            print(f"错误：ChromaDB 知识库目录不存在：{CHROMA_DIR}")
            print("请先运行 python build_kb.py 构建知识库")
            sys.exit(1)

        self.store = VectorStore()
        self.store.load(KB_FILE)

        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def _get_query_embedding(self, query_text):
        """将查询文本转向量"""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query_text],
            dimensions=EMBEDDING_DIMENSIONS,
        )
        import numpy as np
        return np.array(response.data[0].embedding, dtype=np.float32)

    def search(self, query, top_k=5, filter_type=None):
        """
        语义检索

        Args:
            query: 自然语言查询
            top_k: 返回前 K 条
            filter_type: 可选过滤类型
                         dish_profile / dish_attribute / dish_crowd
                         combo_plan / avoid_combo / tips
                         mutual_exclusion / fruit_allergen / level_definition

        Returns:
            list of {id, text, metadata, score}
        """
        query_emb = self._get_query_embedding(query)
        return self.store.search(query_emb, top_k=top_k, filter_type=filter_type)

    def recommend_dishes(self, query, top_k=8):
        """推荐菜品（只搜索菜品档案）"""
        return self.search(query, top_k=top_k, filter_type="dish_profile")

    def get_combo_plans(self, query="聚餐推荐", top_k=3):
        """获取搭配方案"""
        return self.search(query, top_k=top_k, filter_type="combo_plan")

    def get_avoid_combos(self, query="什么不能一起点", top_k=3):
        """获取避雷搭配"""
        return self.search(query, top_k=top_k, filter_type="avoid_combo")

    def get_exclusion_rules(self, query="菌子重复冲突", top_k=3):
        """获取互斥规则"""
        return self.search(query, top_k=top_k, filter_type="mutual_exclusion")

    def get_fruit_allergens(self, query="水果过敏", top_k=5):
        """获取水果过敏原信息"""
        return self.search(query, top_k=top_k, filter_type="fruit_allergen")

    def get_stats(self):
        """获取知识库统计"""
        return self.store.get_stats()

    def list_all_dishes(self):
        """列出所有菜品"""
        return self.store.get_all_dishes()


def format_result(result, index):
    """格式化单条检索结果"""
    meta = result["metadata"]
    score = result["score"]
    text = result["text"]

    # 根据类型选择不同的展示格式
    meta_type = meta.get("type", "")
    header = ""
    if meta_type == "dish_profile":
        dish = meta.get("dish_name", "")
        header = f"  菜品：{dish}"
    elif meta_type == "combo_plan":
        header = f"  {meta.get('section', '')}"
    elif meta_type == "avoid_combo":
        header = f"  避雷：{meta.get('section', '')}"
    elif meta_type == "mutual_exclusion":
        header = f"  互斥：{meta.get('section', '')}"
    elif meta_type == "fruit_allergen":
        header = f"  水果：{meta.get('fruit', '')}"
    else:
        header = f"  [{meta_type}]"

    lines = [
        f"{'─' * 50}",
        f"#{index} {header}  （相似度：{score:.3f}）",
        f"{'─' * 50}",
    ]

    # 展示文本内容（每行缩进）
    for line in text.split("\n"):
        lines.append(f"  {line}")

    # 展示关键元数据
    if meta_type == "dish_profile":
        details = []
        if meta.get("spice_level"):
            details.append(f"辣度={meta['spice_level']}")
        if meta.get("salt_level"):
            details.append(f"咸度={meta['salt_level']}")
        if meta.get("suitable_crowd"):
            details.append(f"适合={meta['suitable_crowd']}")
        if meta.get("allergen_info"):
            details.append(f"过敏原={meta['allergen_info']}")
        if details:
            lines.append(f"  [{', '.join(details)}]")

    return "\n".join(lines)


def interactive_mode():
    """交互式查询模式"""
    print("=" * 60)
    print("菌彩野生菌火锅 · 菜品推荐知识库查询")
    print("=" * 60)

    kb = KnowledgeBaseQuery()
    stats = kb.get_stats()
    print(f"\n知识库已加载：{stats['total']} 条记录，向量维度 {stats['embedding_dim']}")
    print(f"菜品数量：{len(kb.list_all_dishes())} 种")

    print("\n可用命令：")
    print("  直接输入问题进行查询，例如：")
    print("    - 适合老人小孩吃不辣的菜")
    print("    - 有没有微辣的凉菜")
    print("    - 3-4人聚餐推荐什么搭配")
    print("    - 哪些菌子不能一起点")
    print("    - 吃完菌子能吃水果吗")
    print("  输入 'stats' 查看知识库统计")
    print("  输入 'dishes' 列出所有菜品")
    print("  输入 'quit' 退出")

    while True:
        print("\n" + "─" * 60)
        try:
            query = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if query.lower() == "stats":
            print(f"\n总记录数：{stats['total']}")
            print("按类型：")
            for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
                print(f"  {t}: {c}")
            continue
        if query.lower() == "dishes":
            print("\n所有菜品：")
            for i, d in enumerate(kb.list_all_dishes(), 1):
                print(f"  {i:3d}. {d}")
            continue

        # 执行查询：同时搜索菜品档案和搭配方案
        print(f"\n正在检索：'{query}'...")

        # 智能判断查询类型
        if any(kw in query for kw in ["搭配", "推荐", "聚餐", "组合", "方案", "几个人"]):
            # 搭配方案查询
            results = kb.get_combo_plans(query, top_k=3)
            print(f"\n【搭配方案】")
            for i, r in enumerate(results, 1):
                print(format_result(r, i))

            # 同时搜索避雷搭配
            if any(kw in query for kw in ["避雷", "不能", "不要", "冲突"]):
                avoid_results = kb.get_avoid_combos(query, top_k=2)
                if avoid_results:
                    print(f"\n【避雷搭配】")
                    for i, r in enumerate(avoid_results, 1):
                        print(format_result(r, i))

        elif any(kw in query for kw in ["互斥", "重复", "冲突", "不能一起", "相克"]):
            # 互斥规则查询
            results = kb.get_exclusion_rules(query, top_k=5)
            print(f"\n【互斥规则】")
            for i, r in enumerate(results, 1):
                print(format_result(r, i))

        elif any(kw in query for kw in ["水果", "过敏", "芒果", "菠萝"]):
            # 水果过敏查询
            results = kb.get_fruit_allergens(query, top_k=5)
            print(f"\n【水果过敏原信息】")
            for i, r in enumerate(results, 1):
                print(format_result(r, i))

        else:
            # 默认：菜品推荐
            results = kb.recommend_dishes(query, top_k=8)
            print(f"\n【菜品推荐】")
            for i, r in enumerate(results, 1):
                print(format_result(r, i))


def main():
    if len(sys.argv) > 1:
        # 命令行直接查询
        query = " ".join(sys.argv[1:])
        kb = KnowledgeBaseQuery()
        print(f"查询：{query}\n")

        # 菜品推荐
        results = kb.recommend_dishes(query, top_k=5)
        print("【菜品推荐】")
        for i, r in enumerate(results, 1):
            print(format_result(r, i))

        # 搭配方案
        combos = kb.get_combo_plans(query, top_k=2)
        if combos:
            print("\n【搭配方案】")
            for i, r in enumerate(combos, 1):
                print(format_result(r, i))
    else:
        # 交互模式
        interactive_mode()


if __name__ == "__main__":
    main()

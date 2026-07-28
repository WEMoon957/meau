"""菜品知识库查询模块 - 供 Agent 工具调用

从根目录的知识库（kb_data/vector_store.pkl）加载向量数据，
提供语义检索能力，包括：菜品档案、搭配方案、互斥规则、水果过敏原。

本模块作为单例使用，首次调用时自动加载知识库。
"""

import os
import sys

# 将项目根目录加入 sys.path，以便导入 config / kb_store
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config import (
    API_KEY, BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    KB_FILE, CHROMA_DIR,
)
from kb_store import VectorStore

from openai import OpenAI
import numpy as np


# ======================== 单例管理 ========================
_kb_instance = None


def _get_kb() -> VectorStore:
    """获取知识库单例（懒加载）"""
    global _kb_instance
    if _kb_instance is not None:
        return _kb_instance

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB 知识库目录不存在：{CHROMA_DIR}\n"
            "请在项目根目录运行 python build_kb.py 构建知识库"
        )

    _kb_instance = VectorStore()
    _kb_instance.load(KB_FILE)
    return _kb_instance


def _get_client() -> OpenAI:
    """获取 OpenAI 客户端（指向千问端点）"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _get_query_embedding(query_text: str) -> np.ndarray:
    """将查询文本转向量"""
    client = _get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text],
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


# ======================== 预加载 ========================
def preload_kb() -> None:
    """启动时预加载知识库，避免首次检索延迟"""
    try:
        _get_kb()
    except Exception as e:
        print(f"⚠️ 菜品知识库预加载失败: {e}")


# ======================== 查询接口 ========================
def search_dish_knowledge(query: str, top_k: int = 5) -> list[dict]:
    """语义搜索菜品知识（全库检索，不限类型）

    Args:
        query: 自然语言查询
        top_k: 返回条数

    Returns:
        list of {id, text, metadata, score}
    """
    kb = _get_kb()
    emb = _get_query_embedding(query)
    return kb.search(emb, top_k=top_k)


def search_dish_profiles(query: str, top_k: int = 5) -> list[dict]:
    """只搜索菜品档案（辣度/咸度/热量/适合人群/过敏原）

    Args:
        query: 自然语言查询，如"适合老人吃不辣的菜"
        top_k: 返回条数
    """
    kb = _get_kb()
    emb = _get_query_embedding(query)
    return kb.search(emb, top_k=top_k, filter_type="dish_profile")


def search_combo_plans(query: str, top_k: int = 3) -> list[dict]:
    """搜索搭配方案（4套预设套餐）

    Args:
        query: 自然语言查询，如"3-4人聚餐推荐"
        top_k: 返回条数
    """
    kb = _get_kb()
    emb = _get_query_embedding(query)
    return kb.search(emb, top_k=top_k, filter_type="combo_plan")


def search_avoid_combos(query: str, top_k: int = 2) -> list[dict]:
    """搜索避雷搭配（不推荐的组合）

    Args:
        query: 自然语言查询，如"什么不能一起点"
        top_k: 返回条数
    """
    kb = _get_kb()
    emb = _get_query_embedding(query)
    return kb.search(emb, top_k=top_k, filter_type="avoid_combo")


def search_exclusion_rules(query: str, top_k: int = 3) -> list[dict]:
    """搜索菜品互斥规则（菌子重复/口味冲突）

    Args:
        query: 自然语言查询，如"哪些菌子不能一起点"
        top_k: 返回条数
    """
    kb = _get_kb()
    emb = _get_query_embedding(query)
    return kb.search(emb, top_k=top_k, filter_type="mutual_exclusion")


def search_fruit_allergens(query: str, top_k: int = 5) -> list[dict]:
    """搜索水果过敏原信息

    Args:
        query: 自然语言查询，如"吃完菌子能吃芒果吗"
        top_k: 返回条数
    """
    kb = _get_kb()
    emb = _get_query_embedding(query)
    return kb.search(emb, top_k=top_k, filter_type="fruit_allergen")


def get_all_exclusion_rules() -> list[dict]:
    """加载所有菜品互斥规则（不需要向量化，直接批量读取）

    Returns:
        list of {id, text, metadata}
    """
    kb = _get_kb()
    return kb.get_all_by_type("mutual_exclusion")


def get_all_avoid_combos() -> list[dict]:
    """加载所有避雷搭配规则（不需要向量化，直接批量读取）

    Returns:
        list of {id, text, metadata}
    """
    kb = _get_kb()
    return kb.get_all_by_type("avoid_combo")


def get_kb_info() -> dict:
    """获取知识库统计信息"""
    kb = _get_kb()
    return kb.get_stats()


def list_all_dishes() -> list[str]:
    """列出知识库中所有菜品名称"""
    kb = _get_kb()
    return kb.get_all_dishes()

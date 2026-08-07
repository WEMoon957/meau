"""向量库管理模块 - 使用 Chroma + 阿里云 Embedding 构建话术向量库

功能：
1. 构建向量库：将话术数据向量化并存入 Chroma
2. 语义检索：根据查询检索最相关的话术
3. 手动补充：支持添加自定义话术到向量库
4. 持久化：向量库保存在本地磁盘，重启不丢失
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from script_data import generate_all_scripts, add_custom_script


# 向量库存储路径
VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vector_db")
COLLECTION_NAME = "server_scripts"

# Embedding 模型配置
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v2")

_embedding_instance = None
_vector_store_instance = None


def _get_embedding():
    """获取 Embedding 模型实例（单例，避免重复创建客户端）"""
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    from langchain_core.embeddings import Embeddings
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    class DashScopeEmbeddings(Embeddings):
        """阿里云 DashScope 兼容的 Embedding 实现"""

        def __init__(self, model: str, api_key: str, base_url: str = ""):
            self.model = model
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # 分批处理，每批最多 25 条（DashScope 限制）
            all_embeddings = []
            batch_size = 25
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self.client.embeddings.create(
                    model=self.model, input=batch
                )
                all_embeddings.extend([d.embedding for d in resp.data])
            return all_embeddings

        def embed_query(self, text: str) -> list[float]:
            resp = self.client.embeddings.create(
                model=self.model, input=text
            )
            return resp.data[0].embedding

    _embedding_instance = DashScopeEmbeddings(
        model=EMBEDDING_MODEL, api_key=api_key, base_url=base_url
    )
    return _embedding_instance


def build_vector_store(scripts: list[dict] = None, force_rebuild: bool = False):
    """构建（或加载）向量库

    Args:
        scripts: 话术数据列表，为None则自动生成
        force_rebuild: 是否强制重建（删除旧数据重新构建）

    Returns:
        Chroma 向量库实例
    """
    global _vector_store_instance
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    embedding = _get_embedding()

    # 如果强制重建，先删除旧数据
    if force_rebuild and os.path.exists(VECTOR_DB_PATH):
        import shutil
        shutil.rmtree(VECTOR_DB_PATH)
        _vector_store_instance = None

    # 优先返回已加载的实例
    if not force_rebuild and _vector_store_instance is not None:
        return _vector_store_instance

    # 如果向量库已存在，直接加载
    if not force_rebuild and os.path.exists(VECTOR_DB_PATH):
        _vector_store_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=VECTOR_DB_PATH,
        )
        return _vector_store_instance

    # 生成话术数据
    if scripts is None:
        scripts = generate_all_scripts()

    # 转换为 Document
    documents = []
    for s in scripts:
        doc = Document(
            page_content=s["content"],
            metadata={
                "script_id": s["id"],
                "type": s["type"],
                "dish_name": s.get("dish_name", ""),
                **s.get("metadata", {}),
            }
        )
        documents.append(doc)

    # 构建向量库
    _vector_store_instance = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
    )

    return _vector_store_instance


def preload_vector_store() -> None:
    """启动时预加载向量库，避免首次检索延迟"""
    build_vector_store()


def search_scripts(query: str, k: int = 3, script_type: str = None) -> list[dict]:
    """语义检索话术

    Args:
        query: 查询文本，如"顾客带小孩来吃什么"
        k: 返回条数
        script_type: 限定话术类型（selling_point/scene/pairing/exception/custom），为None则不限

    Returns:
        匹配的话术列表，每条包含 content, score, metadata
    """
    vector_store = build_vector_store()

    # 构建过滤条件
    filter_dict = None
    if script_type:
        filter_dict = {"type": script_type}

    results = vector_store.similarity_search_with_relevance_scores(
        query, k=k, filter=filter_dict
    )

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "score": round(score, 3),
            "metadata": doc.metadata,
        })
    return output


def add_script(content: str, script_type: str = "custom",
               dish_name: str = "", scene: str = "") -> str:
    """添加自定义话术到向量库

    Args:
        content: 话术内容
        script_type: 话术类型
        dish_name: 关联菜品名（可选）
        scene: 场景描述（可选）

    Returns:
        添加结果提示
    """
    from langchain_core.documents import Document

    script = add_custom_script(content, script_type, dish_name, scene)

    vector_store = build_vector_store()
    doc = Document(
        page_content=script["content"],
        metadata={
            "script_id": script["id"],
            "type": script["type"],
            "dish_name": script.get("dish_name", ""),
            **script.get("metadata", {}),
        }
    )
    vector_store.add_documents([doc])

    return f"已添加自定义话术（ID: {script['id']}）"


# ======================== 异步接口（供 async 工具调用） ========================
async def asearch_scripts(query: str, k: int = 3, script_type: str = None) -> list[dict]:
    """search_scripts 的异步版本

    Chroma 的同步 API 用 to_thread 包装，避免阻塞事件循环。
    Embedding 调用（DashScope 远程 API）是真正的 I/O 等待点。
    """
    def _search():
        vector_store = build_vector_store()

        filter_dict = None
        if script_type:
            filter_dict = {"type": script_type}

        results = vector_store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )

        output = []
        for doc, score in results:
            output.append({
                "content": doc.page_content,
                "score": round(score, 3),
                "metadata": doc.metadata,
            })
        return output

    return await asyncio.to_thread(_search)


async def aadd_script(content: str, script_type: str = "custom",
                     dish_name: str = "", scene: str = "") -> str:
    """add_script 的异步版本"""
    def _add():
        from langchain_core.documents import Document
        script = add_custom_script(content, script_type, dish_name, scene)

        vector_store = build_vector_store()
        doc = Document(
            page_content=script["content"],
            metadata={
                "script_id": script["id"],
                "type": script["type"],
                "dish_name": script.get("dish_name", ""),
                **script.get("metadata", {}),
            }
        )
        vector_store.add_documents([doc])

        return f"已添加自定义话术（ID: {script['id']}）"

    return await asyncio.to_thread(_add)


def get_vector_store_info() -> dict:
    """获取向量库信息"""
    vector_store = build_vector_store()
    collection = vector_store._collection
    count = collection.count()

    # 统计各类型数量
    type_counts = {}
    if count > 0:
        results = collection.get(include=["metadatas"])
        for meta in results.get("metadatas", []):
            t = meta.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total": count,
        "type_counts": type_counts,
        "db_path": VECTOR_DB_PATH,
    }


if __name__ == "__main__":
    # 构建向量库
    print("正在构建向量库...")
    vs = build_vector_store(force_rebuild=True)

    info = get_vector_store_info()
    print(f"\n向量库构建完成！")
    print(f"  总条数: {info['total']}")
    print(f"  存储路径: {info['db_path']}")
    print(f"  类型分布:")
    for t, c in info["type_counts"].items():
        print(f"    {t}: {c} 条")

    # 测试检索
    print("\n" + "=" * 60)
    print("检索测试")
    print("=" * 60)

    test_queries = [
        "顾客带小孩来吃什么好",
        "推荐辣菜",
        "顾客说菜太辣了怎么办",
        "四个人聚餐怎么点",
    ]

    for q in test_queries:
        print(f"\n查询: {q}")
        results = search_scripts(q, k=2)
        for r in results:
            print(f"  [{r['score']}] {r['content'][:80]}...")

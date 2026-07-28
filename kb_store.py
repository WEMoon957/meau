"""
菌彩野生菌火锅 · 菜品推荐知识库 - 向量存储模块（ChromaDB 生产级实现）

基于 ChromaDB PersistentClient 的持久化向量存储，特性：
- HNSW 索引 + cosine 距离，检索性能远优于 numpy 暴力计算
- 元数据字段化存储，支持 where 过滤（type / source 等）
- 进程级持久化，写入即落盘，无需显式 dump
- 多进程/多实例共享同一知识库目录
- 接口与旧版 numpy 实现完全兼容：add / search / save / load / count
  以及 get_all_dishes / get_all_by_type / get_stats

注意：本文件命名为 kb_store.py，避免与 main/vector_store.py（话术向量库）冲突。
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings

# 从 config 读取默认 ChromaDB 配置（避免循环导入：config 不依赖本模块）
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    CHROMA_DISTANCE,
    CHROMA_BATCH_SIZE,
)


class VectorStore:
    """基于 ChromaDB 的生产级向量存储

    接口与旧版 numpy 实现保持一致：
      - add(ids, documents, metadatas, embeddings)
      - search(query_embedding, top_k, filter_type, filter_source)
      - save(path) / load(path)  # path 仅作兼容参数，实际使用 CHROMA_DIR
      - count() / get_all_dishes() / get_all_by_type() / get_stats()
    """

    # ChromaDB 元数据字段允许的值类型；其他类型自动 str() 转换
    _SCALAR_TYPES = (str, int, float, bool)

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance: Optional[str] = None,
    ):
        """初始化向量存储（惰性连接，首次 add/save/load 时才创建客户端）

        Args:
            persist_dir: ChromaDB 持久化目录，默认使用 config.CHROMA_DIR
            collection_name: 集合名，默认使用 config.CHROMA_COLLECTION
            distance: 距离度量 cosine/l2/ip，默认使用 config.CHROMA_DISTANCE
        """
        self._persist_dir = persist_dir or CHROMA_DIR
        self._collection_name = collection_name or CHROMA_COLLECTION
        self._distance = distance or CHROMA_DISTANCE

        self._client: Optional[chromadb.api.ClientAPI] = None
        self._collection = None

        # 内存缓存：用于 get_all_dishes / get_all_by_type / get_stats 等
        # 批量遍历场景，避免每次都走 ChromaDB 全量查询
        self._ids_cache: list[str] = []
        self._documents_cache: list[str] = []
        self._metadatas_cache: list[dict] = []
        self._embedding_dim: int = 0

    # ------------------------------------------------------------------
    # 内部：连接 / 集合管理
    # ------------------------------------------------------------------
    def _ensure_client(self):
        """惰性初始化 PersistentClient"""
        if self._client is not None:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    def _ensure_collection(self, create_if_missing: bool = True):
        """确保集合已打开

        Args:
            create_if_missing: True=不存在则创建；False=不存在抛错
        """
        if self._collection is not None:
            return
        self._ensure_client()
        if create_if_missing:
            # cosine 距离通过 collection metadata 配置；HNSW 为默认索引
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": self._distance},
            )
        else:
            try:
                self._collection = self._client.get_collection(
                    name=self._collection_name
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"ChromaDB 集合不存在：{self._collection_name}\n"
                    f"目录：{self._persist_dir}\n"
                    f"请先运行 python build_kb.py 构建知识库\n"
                    f"原始错误：{e}"
                ) from e

    @classmethod
    def _clean_metadata(cls, meta: dict) -> dict:
        """清洗 metadata，确保 ChromaDB 兼容

        - 跳过 None 值（ChromaDB 不允许 metadata 值为 None）
        - list/dict 等非标量自动 str() 序列化
        """
        clean = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, cls._SCALAR_TYPES):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _to_list(vec):
        """numpy array / tuple / list 统一转为 list[float]"""
        if vec is None:
            return None
        if hasattr(vec, "tolist"):
            return vec.tolist()
        if isinstance(vec, list):
            return vec
        return list(vec)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(self, ids, documents, metadatas, embeddings):
        """批量 upsert 文档与向量

        幂等：相同 id 会覆盖，便于知识库增量重建。
        """
        self._ensure_collection(create_if_missing=True)

        if not ids:
            return

        # 统一类型
        ids = [str(i) for i in ids]
        embeddings = [self._to_list(e) for e in embeddings]
        metadatas = [self._clean_metadata(m) for m in metadatas]

        # 记录向量维度
        if embeddings and self._embedding_dim == 0:
            self._embedding_dim = len(embeddings[0])

        # 分批 upsert，避免单次请求过大
        n = len(ids)
        for start in range(0, n, CHROMA_BATCH_SIZE):
            end = start + CHROMA_BATCH_SIZE
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        # 维护内存缓存（追加，不覆盖；upsert 已在 ChromaDB 内去重）
        # 为保证缓存一致性，这里简单地全量重载
        self._reload_cache()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding,
        top_k: int = 5,
        filter_type: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> list[dict]:
        """语义检索

        Args:
            query_embedding: 查询向量（numpy array / list 均可）
            top_k: 返回前 K 条
            filter_type: 按元数据 type 过滤
            filter_source: 按元数据 source 过滤

        Returns:
            list of {id, text, metadata, score}  score 为 [0,1] 的相似度
        """
        if self._collection is None:
            return []
        try:
            if self._collection.count() == 0:
                return []
        except Exception:
            return []

        query_emb = self._to_list(query_embedding)
        if query_emb is None:
            return []

        # 构建 ChromaDB where 子句
        where = self._build_where(filter_type, filter_source)

        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        try:
            res = self._collection.query(**kwargs)
        except Exception as e:
            print(f"⚠️ ChromaDB 查询失败：{e}")
            return []

        # ChromaDB 返回结构：每个字段都是 list[list[...]]（外层=查询数）
        ids_batch = res.get("ids", [[]])
        docs_batch = res.get("documents", [[]])
        metas_batch = res.get("metadatas", [[]])
        dists_batch = res.get("distances", [[]])

        if not ids_batch:
            return []

        ids = ids_batch[0]
        docs = docs_batch[0] if docs_batch else []
        metas = metas_batch[0] if metas_batch else []
        dists = dists_batch[0] if dists_batch else []

        out = []
        for i, _id in enumerate(ids):
            distance = dists[i] if i < len(dists) else 1.0
            # cosine 距离取值 [0, 2]，相似度 = 1 - distance，截断到 [0, 1]
            score = max(0.0, 1.0 - float(distance))
            out.append({
                "id": _id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": score,
            })
        return out

    @staticmethod
    def _build_where(filter_type, filter_source):
        """构建 ChromaDB where 过滤条件"""
        clauses = []
        if filter_type:
            clauses.append({"type": filter_type})
        if filter_source:
            clauses.append({"source": filter_source})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # ------------------------------------------------------------------
    # 持久化（接口兼容）
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None):
        """保存到 ChromaDB（持久化客户端，写入即落盘）

        path 参数仅为兼容旧接口，实际持久化目录由 CHROMA_DIR 决定。
        """
        self._ensure_collection(create_if_missing=True)
        print(
            f"知识库已保存到 ChromaDB：{self._persist_dir}"
            f"（collection={self._collection_name}，{self.count()} 条记录）"
        )

    def load(self, path: Optional[str] = None):
        """从 ChromaDB 加载集合

        path 参数仅为兼容旧接口，实际持久化目录由 CHROMA_DIR 决定。
        """
        if not os.path.exists(self._persist_dir):
            raise FileNotFoundError(
                f"ChromaDB 目录不存在：{self._persist_dir}\n"
                "请先运行 python build_kb.py 构建知识库"
            )
        self._ensure_client()
        self._ensure_collection(create_if_missing=False)
        self._reload_cache()
        print(
            f"知识库已从 ChromaDB 加载：{self._persist_dir}"
            f"（{self.count()} 条记录，向量维度 {self._embedding_dim}）"
        )

    def _reload_cache(self):
        """从 ChromaDB 全量加载到内存缓存（用于统计/批量遍历）"""
        if self._collection is None:
            return
        try:
            data = self._collection.get(
                include=["documents", "metadatas"]
            )
            self._ids_cache = list(data.get("ids", []))
            self._documents_cache = list(data.get("documents", []))
            self._metadatas_cache = list(data.get("metadatas", []))
        except Exception as e:
            print(f"⚠️ 加载 ChromaDB 缓存失败：{e}")
            self._ids_cache = []
            self._documents_cache = []
            self._metadatas_cache = []
            return

        # 通过 peek 探测向量维度（ChromaDB 1.5+ 的 peek 不接受 include，
        # 默认即返回 embeddings；返回的 embeddings 是 numpy array，
        # 不能用 `or []`，否则多元素数组会抛 ValueError）
        try:
            peek = self._collection.peek(limit=1)
            embs = peek.get("embeddings")
            if embs is not None and len(embs) > 0:
                self._embedding_dim = len(embs[0])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 统计 / 批量读取
    # ------------------------------------------------------------------
    def count(self) -> int:
        """返回集合中的文档数"""
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return len(self._ids_cache)

    def get_all_dishes(self) -> list[str]:
        """获取所有菜品名称（去重排序）"""
        dishes = set()
        for meta in self._metadatas_cache:
            name = meta.get("dish_name")
            if name:
                dishes.add(name)
        return sorted(dishes)

    def get_all_by_type(self, type_name: str) -> list[dict]:
        """按 type 批量加载所有条目（无需向量化检索）

        Args:
            type_name: 元数据 type 值，如 'mutual_exclusion'、'avoid_combo'

        Returns:
            list of {id, text, metadata}
        """
        # 优先走 ChromaDB where 查询，缓存仅作兜底
        if self._collection is not None:
            try:
                res = self._collection.get(
                    where={"type": type_name},
                    include=["documents", "metadatas"],
                )
                ids = res.get("ids", [])
                docs = res.get("documents", [])
                metas = res.get("metadatas", [])
                return [
                    {
                        "id": ids[i],
                        "text": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                    }
                    for i in range(len(ids))
                ]
            except Exception as e:
                print(f"⚠️ ChromaDB where 查询失败，回退到内存缓存：{e}")

        # 兜底：从内存缓存过滤
        results = []
        for i in range(len(self._documents_cache)):
            meta = self._metadatas_cache[i] if i < len(self._metadatas_cache) else {}
            if meta.get("type") == type_name:
                results.append({
                    "id": self._ids_cache[i] if i < len(self._ids_cache) else "",
                    "text": self._documents_cache[i],
                    "metadata": meta,
                })
        return results

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        type_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for meta in self._metadatas_cache:
            t = meta.get("type", "unknown")
            s = meta.get("source", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            source_counts[s] = source_counts.get(s, 0) + 1
        return {
            "total": self.count(),
            "by_type": type_counts,
            "by_source": source_counts,
            "embedding_dim": self._embedding_dim,
        }

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    def reset(self):
        """清空当前持久化目录下的所有集合（谨慎使用）"""
        self._ensure_client()
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = None
        self._ids_cache = []
        self._documents_cache = []
        self._metadatas_cache = []
        self._embedding_dim = 0
        print(f"已清空集合：{self._collection_name}")

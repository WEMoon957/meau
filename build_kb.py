"""
菌彩野生菌火锅 · 菜品推荐知识库 - 构建脚本

使用流程：
  1. 编辑 config.py，填入你的千问 API Key
  2. 运行：python build_kb.py
  3. 知识库将保存到 kb_data/vector_store.pkl
"""

import sys
import time
import os

# 确保能导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    API_KEY, BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    BATCH_SIZE, KB_FILE, CHROMA_DIR,
)
from data_processor import process_all
from kb_store import VectorStore
from openai import OpenAI


def get_embeddings(texts, client):
    """
    调用千问 Embedding API 获取向量

    Args:
        texts: 文本列表
        client: OpenAI 客户端（指向千问端点）

    Returns:
        list of float arrays
    """
    all_embeddings = []
    total = len(texts)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = texts[i:i + BATCH_SIZE]
        print(f"  向量化中... 批次 {batch_num}/{total_batches}（{len(batch)} 条）", end="", flush=True)

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f" -> 完成（{response.usage.total_tokens} tokens）")
        except Exception as e:
            print(f" -> 失败：{e}")
            raise

        # 避免API限流
        if i + BATCH_SIZE < total:
            time.sleep(0.5)

    return all_embeddings


def main():
    print("=" * 60)
    print("菌彩野生菌火锅 · 菜品推荐知识库构建")
    print("=" * 60)

    # 1. 检查 API Key
    if "your-dashscope-api-key" in API_KEY:
        print("\n错误：请先在 config.py 中填入你的千问 API Key！")
        print("获取地址：https://dashscope.console.aliyun.com/apiKey")
        sys.exit(1)

    # 2. 解析数据文件
    print("\n[1/3] 解析数据文件...")
    chunks = process_all()

    # 3. 调用千问 API 向量化
    print(f"\n[2/3] 调用千问 Embedding API 向量化（模型：{EMBEDDING_MODEL}）...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = get_embeddings(texts, client)
    print(f"  向量化完成，共 {len(embeddings)} 条，维度 {len(embeddings[0])}")

    # 4. 存入向量库（ChromaDB）
    print(f"\n[3/3] 存入向量库（ChromaDB：{CHROMA_DIR}）...")
    store = VectorStore()
    # 重建前清空旧集合，避免残留脏数据（id 变更时旧记录不会被自动删除）
    store.reset()
    store.add(ids, texts, metadatas, embeddings)
    store.save(KB_FILE)

    # 清理遗留的旧版 numpy pkl 文件（若存在）
    if os.path.exists(KB_FILE):
        try:
            os.remove(KB_FILE)
            print(f"  已清理旧版向量库文件：{KB_FILE}")
        except OSError:
            pass

    # 5. 打印统计信息
    print("\n" + "=" * 60)
    print("知识库构建完成！")
    print("=" * 60)
    stats = store.get_stats()
    print(f"总记录数：{stats['total']}")
    print(f"向量维度：{stats['embedding_dim']}")
    print(f"\n按类型统计：")
    for t, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {count} 条")
    print(f"\n按来源统计：")
    for s, count in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {s}: {count} 条")

    print(f"\nChromaDB 目录：{CHROMA_DIR}")
    print(f"下一步：运行 python query.py 进行菜品推荐查询")


if __name__ == "__main__":
    main()

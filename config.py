"""
菌彩野生菌火锅 · 菜品推荐知识库 - 配置文件

API Key 获取方式（二选一）：
  方式一：在项目根目录的 .env 文件中配置（与现有项目共用）
    OPENAI_API_KEY=sk-your-dashscope-api-key
    OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    EMBEDDING_MODEL=text-embedding-v3

  方式二：直接修改下方 API_KEY 变量

  API Key 获取地址：https://dashscope.console.aliyun.com/apiKey

使用流程：
  1. 配置好 API Key 后，运行 python build_kb.py 构建知识库
  2. 运行 python query.py 进行菜品推荐查询
"""

import os

# ============================================================
# 加载 .env 文件（如果存在）
# ============================================================
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# ============================================================
# 千问 Embedding API 配置（独立于对话 LLM）
# ============================================================
API_KEY = os.environ.get(
    "EMBEDDING_API_KEY",
    os.environ.get("OPENAI_API_KEY", "sk-your-dashscope-api-key-here"),
)

BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL",
    os.environ.get(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

# 向量化模型：text-embedding-v3 支持中英文，效果优秀
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")

# 向量维度（text-embedding-v3 支持 1024 或 768，推荐 1024）
EMBEDDING_DIMENSIONS = 1024

# 每次API请求的最大文本数（Workspace Key 限制单批不超过 10 条）
BATCH_SIZE = 10

# ============================================================
# 知识库存储路径
# ============================================================
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_data")
KB_FILE = os.path.join(KB_DIR, "vector_store.pkl")

# ============================================================
# ChromaDB 配置（生产级向量库）
# ============================================================
# ChromaDB 持久化目录（数据存放在此目录下的 chroma.sqlite3 + 集合文件）
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(KB_DIR, "chroma_db"))
# 集合名称（一个持久化目录可同时承载多个集合）
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "dish_kb")
# 距离度量：cosine / l2 / ip（千问 text-embedding-v3 已做归一化，cosine 最合适）
CHROMA_DISTANCE = os.environ.get("CHROMA_DISTANCE", "cosine")
# upsert 单批大小（ChromaDB 推荐 <= 5000，这里保守取 500 以兼顾内存与速度）
CHROMA_BATCH_SIZE = int(os.environ.get("CHROMA_BATCH_SIZE", "500"))

# ============================================================
# 原始数据文件路径
# ============================================================
ATTACH_DIR = r"c:\Users\work\.trae-cn\attachments\6a68076bbf3e3e8c2cdedd89"

DATA_FILES = {
    "spice_salt_calorie": os.path.join(
        ATTACH_DIR,
        "a8de9b6b-24ad-4fed-95e9-f397354caa33_ee934a73-2286-46df-8c9f-7f600eb9de4c_菌彩-菜品辣度咸度分级表.xlsx",
    ),
    "fruit_allergen": os.path.join(
        ATTACH_DIR,
        "228416a0-ace7-42bd-8888-f7e750ba5f1d_f2be5945-0283-4c68-b27b-1773816b2685_水果过敏原信息.xlsx",
    ),
    "suitable_crowd": os.path.join(
        ATTACH_DIR,
        "0a26c8db-1bd5-4d18-98e2-56a5915a3639_9daf053f-79dd-4c9c-8a80-884245627d4b_云南菌彩野生菌火锅适合人群.xls",
    ),
    "dish_pairing": os.path.join(
        ATTACH_DIR,
        "1119de9c-f5a5-4a7a-8ceb-3bba5d44eda9_e1a2136f-8240-4408-8dcd-6669f7529fbb_菜品搭配关系.docx",
    ),
    "mutual_exclusion": os.path.join(
        ATTACH_DIR,
        "aac1491c-7081-4f26-b770-e91481b6c4d2_29ed76f3-a2a7-49c5-88c5-397ad03c377f_13项菌子容易重复、口味冲突.docx",
    ),
}

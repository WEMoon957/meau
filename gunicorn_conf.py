"""gunicorn 配置 - 多 worker 高并发部署

启动方式（生产）：
    gunicorn -c gunicorn_conf.py main.api_server:app

设计要点：
  - worker_class=uvicorn.workers.UvicornWorker：让 FastAPI 的 async + lifespan 在每个 worker 内正常运行。
  - preload_app=False（默认）：每个 worker 独立导入应用并独立运行 lifespan，
    从而各自创建自己的 MySQL 连接池 / Redis 客户端 / 知识库单例，
    避免 fork 后共享 MySQL 连接（pymysql 非进程安全）。
  - max_requests + jitter：定期回收 worker，缓解 LangChain/ChromaDB 长期运行的内存碎片。
  - graceful_timeout：收到 SIGTERM 后等待 in-flight 请求处理完的最大时长，
    与 LLM_REQUEST_TIMEOUT 协调（需 >= LLM 超时 + 余量），保证优雅关停不切断正在进行的对话。
  - workers：通过 WORKERS 环境变量配置；需保证 DB_POOL_SIZE × WORKERS <= MySQL max_connections × 0.8。
"""

import os
import multiprocessing

# 绑定地址：仅本机，由 nginx 反代到公网
bind = "127.0.0.1:3000"

# worker 数：环境变量优先，否则 CPU*2+1
workers = int(os.environ.get("WORKERS", str(multiprocessing.cpu_count() * 2 + 1)))

# UvicornWorker：保留 FastAPI async 与 lifespan 语义
worker_class = "uvicorn.workers.UvicornWorker"

# 单次请求超时（含 LLM 多轮工具调用）。需 >= LLM_REQUEST_TIMEOUT + 余量
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "60"))

# 优雅关停等待时长：>= timeout，保证 in-flight 请求能跑完
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "75"))

# 长连接 keepalive（与 nginx upstream keepalive 配合）
keepalive = 5

# 定期回收 worker，缓解内存碎片（每个 worker 处理 N 个请求后重启）
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# 不预加载应用：确保每个 worker 独立持有 MySQL/Redis/KB 连接（fork 安全）
preload_app = False

# 日志
accesslog = os.environ.get("GUNICORN_ACCESSLOG", "-")
errorlog = os.environ.get("GUNICORN_ERRORLOG", "-")
loglevel = os.environ.get("LOG_LEVEL", "info").lower()


def worker_exit(server, worker):
    """worker 退出时尽量清理资源（DB 池由 lifespan 兜底关闭，此处为双保险）。"""
    try:
        import sys
        if "main.db" in sys.modules:
            sys.modules["main.db"].close_pool()
    except Exception:
        pass

# 菌彩点餐智能体 - 生产镜像
# 运行：gunicorn + UvicornWorker（保留 FastAPI async + lifespan）
# 依赖外部 MySQL 与 Redis（通过环境变量连接，不在镜像内）

FROM python:3.11-slim

# 国内镜像源加速（apt 用阿里云 Debian 源，pip 用清华 PyPI 源）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null \
    || true

# 系统依赖：
#   - libmysqlclient-dev / pkg-config：pymysql 编译/运行
#   - default-libmysqlclient-dev：DBUtils 直连 MySQL 客户端
#   - build-essential：cryptography/ChromaDB 编译
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        default-libmysqlclient-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 复制项目代码（.dockerignore 会排除 .env / __pycache__ / kb_data 等）
COPY . .

# 暴露服务端口（与 gunicorn_conf.py 的 bind 端口一致）
EXPOSE 3000

# 健康检查（命中 /api/health，含 DB/Redis/KB 探活）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:3000/api/health', timeout=3); sys.exit(0)" || exit 1

# gunicorn 启动：
#   - 绑定 0.0.0.0:3000（覆盖 gunicorn_conf.py 的 127.0.0.1，让容器外可访问）
#   - 不预加载应用（preload_app=False）：每个 worker 独立持有 MySQL/Redis/KB 连接
#   - worker 数由 WORKERS 环境变量控制
CMD ["gunicorn", "-c", "gunicorn_conf.py", "-b", "0.0.0.0:3000", "main.api_server:app"]

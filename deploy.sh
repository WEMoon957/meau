#!/bin/bash
set -e

APP_NAME="menu_recommendation"
APP_DIR="/opt/menu_recommendation"
USER="www-data"
PYTHON_VERSION="3.11"

read -p "请输入 OPENAI_API_KEY: " API_KEY
read -p "请输入 MySQL root 密码: " DB_ROOT_PASSWORD
read -p "请输入数据库密码: " DB_PASSWORD
read -p "请输入小程序前端 CORS 域名(逗号分隔，如 https://order.example.com): " CORS_ORIGINS

echo ""
echo "=========================================="
echo "  小菌点餐智能体 - 阿里云部署脚本（高并发版）"
echo "=========================================="

echo "[1/8] 更新系统并安装依赖..."
apt-get update -y && apt-get upgrade -y
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    nginx git mysql-server libmysqlclient-dev build-essential redis-server

echo "[2/8] 配置 MySQL（提高 max_connections 以容纳 多worker×连接池）..."
mysql -u root <<EOF
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '${DB_ROOT_PASSWORD}';
FLUSH PRIVILEGES;
CREATE DATABASE IF NOT EXISTS restaurant CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'menu_user'@'localhost' IDENTIFIED BY '${DB_PASSWORD}';
GRANT ALL PRIVILEGES ON restaurant.* TO 'menu_user'@'localhost';
FLUSH PRIVILEGES;
EOF
# 连接池×worker 默认 20×4=80，留余量设 200
sed -i "s/^max_connections.*/max_connections = 200/" /etc/mysql/mysql.conf.d/mysqld.cnf || \
    echo "max_connections = 200" >> /etc/mysql/mysql.conf.d/mysqld.cnf
systemctl restart mysql
echo "✅ MySQL 配置完成（max_connections=200）"

echo "[3/8] 启动 Redis（多 worker 共享会话/限流的前提）..."
systemctl enable redis-server
systemctl start redis-server
# 绑定本机即可（应用与 Redis 同机）
echo "✅ Redis 已启动（127.0.0.1:6379）"

echo "[4/8] 创建应用目录并拉取代码..."
mkdir -p ${APP_DIR}
chown ${USER}:${USER} ${APP_DIR}
cd ${APP_DIR}
if [ -d ".git" ]; then
    sudo -u ${USER} git pull origin main
else
    sudo -u ${USER} git clone https://github.com/WEMoon957/meau.git .
fi

echo "[5/8] 创建虚拟环境并安装依赖..."
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[6/8] 初始化数据库与配置..."
cd ${APP_DIR}/main
source ${APP_DIR}/venv/bin/activate
python init_db.py

# CPU 核心数×2+1
WORKERS=$(nproc)
if [ "${WORKERS}" -gt 8 ]; then WORKERS=8; fi
# 保证 DB_POOL_SIZE(20) × WORKERS < MySQL max_connections(200) × 0.8 = 160
cat > ${APP_DIR}/.env <<EOF
OPENAI_API_KEY=${API_KEY}
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo
USE_DATABASE=true
DB_HOST=localhost
DB_PORT=3306
DB_USER=menu_user
DB_PASSWORD=${DB_PASSWORD}
DB_NAME=restaurant
REDIS_URL=redis://127.0.0.1:6379/0
WORKERS=${WORKERS}
MAX_CONCURRENT_CHATS=20
LLM_REQUEST_TIMEOUT=30
DB_POOL_SIZE=20
DB_MAX_EXECUTION_MS=5000
SESSION_TTL_SECONDS=1800
SESSION_CLEANUP_INTERVAL=300
MAX_SESSIONS=500
CHAT_RATE_PER_SESSION=30
CHAT_RATE_PER_IP=60
CHAT_RATE_WINDOW=60
CORS_ALLOWED_ORIGINS=${CORS_ORIGINS}
TRUSTED_PROXIES=127.0.0.1/32
LOG_LEVEL=INFO
EOF

echo "[7/8] 配置 gunicorn（多 worker systemd 服务）..."
cat > /etc/systemd/system/${APP_NAME}.service <<EOF
[Unit]
Description=Menu Recommendation API Service (gunicorn multi-worker)
After=network.target mysql.service redis-server.service

[Service]
Type=notify
User=${USER}
WorkingDirectory=${APP_DIR}
Environment="PYTHONPATH=${APP_DIR}"
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/gunicorn -c ${APP_DIR}/gunicorn_conf.py main.api_server:app
ExecReload=/bin/kill -s HUP \$MAINPID
KillSignal=SIGTERM
Restart=always
RestartSec=5
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${APP_NAME}
systemctl restart ${APP_NAME}

echo "[8/8] 配置 nginx（upstream keepalive + 连接限额 + 探活摘流）..."
# 在 http 块加入 limit_conn_zone（幂等追加，避免重复）
if ! grep -q "limit_conn_zone" /etc/nginx/nginx.conf; then
    sed -i '/http {/a\    limit_conn_zone \$binary_remote_addr zone=perip:10m;' /etc/nginx/nginx.conf
fi

cat > /etc/nginx/sites-available/${APP_NAME} <<EOF
upstream menu_backend {
    server 127.0.0.1:3000;
    keepalive 32;
}

server {
    listen 80;
    server_name _;

    # 单 IP 并发连接上限，防止恶意/异常客户端打满 worker
    limit_conn perip 50;

    # 请求体大小限制（聊天消息较短，2KB 足够）
    client_max_body_size 4k;

    # 主动健康检查的探活端点（nginx 商业版有 active check，这里用被动 + LB 外部探活）
    location = /healthz {
        access_log off;
        proxy_pass http://menu_backend/api/health;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    location / {
        proxy_pass http://menu_backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        # 与 upstream keepalive 配合，复用后端连接
        proxy_set_header Connection "";

        # 与 LLM_REQUEST_TIMEOUT(30s) + gunicorn timeout(60s) 对齐
        proxy_connect_timeout 5s;
        proxy_read_timeout 75s;
        proxy_send_timeout 30s;

        # 缓冲响应，慢客户端不占用后端连接
        proxy_buffering on;
        proxy_buffers 16 8k;
        proxy_busy_buffers_size 16k;
    }
}
EOF

ln -sf /etc/nginx/sites-available/${APP_NAME} /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

echo ""
echo "=========================================="
echo "  部署完成！"
echo "=========================================="
echo ""
echo "  worker 数: ${WORKERS}"
echo "  每worker并发上限: 20  → 总并发 LLM 调用: $((WORKERS * 20))"
echo "  MySQL max_connections: 200（DB_POOL_SIZE×WORKERS=$((20*WORKERS))，留余量）"
echo "  Redis: 127.0.0.1:6379（会话+限流共享）"
echo ""
echo "服务状态:"
systemctl status ${APP_NAME} --no-pager | head -n 15
echo ""
echo "访问地址: http://你的服务器IP/api/health"
echo ""
echo "后续操作:"
echo "1. 在阿里云安全组放行 80 端口（入方向）"
echo "2. 查看日志: journalctl -u ${APP_NAME} -f"
echo "3. 重启服务: systemctl restart ${APP_NAME}"
echo "4. 平滑 reload（不中断连接）: systemctl reload ${APP_NAME}"
echo "5. 配置 HTTPS（可选）: certbot --nginx"
echo ""
echo "容量校验（高并发前必看）:"
echo "  - 中午峰值预估并发用户数 若超过 \${WORKERS}×20，需调大 WORKERS 或 MAX_CONCURRENT_CHATS"
echo "  - 同步调整 MySQL max_connections 确保 >= DB_POOL_SIZE × WORKERS / 0.8"

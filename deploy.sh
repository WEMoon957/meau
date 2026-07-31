#!/bin/bash
set -e

APP_NAME="menu_recommendation"
APP_DIR="/opt/menu_recommendation"
USER="www-data"
PYTHON_VERSION="3.11"

read -p "请输入 OPENAI_API_KEY: " API_KEY
read -p "请输入 MySQL root 密码: " DB_ROOT_PASSWORD
read -p "请输入数据库密码: " DB_PASSWORD

echo ""
echo "=========================================="
echo "  小味点餐智能体 - 阿里云部署脚本"
echo "=========================================="

echo "[1/6] 更新系统并安装依赖..."
apt-get update -y && apt-get upgrade -y
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    nginx git mysql-server libmysqlclient-dev build-essential

echo "[2/6] 配置 MySQL..."
mysql -u root <<EOF
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '${DB_ROOT_PASSWORD}';
FLUSH PRIVILEGES;
CREATE DATABASE IF NOT EXISTS restaurant CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'menu_user'@'localhost' IDENTIFIED BY '${DB_PASSWORD}';
GRANT ALL PRIVILEGES ON restaurant.* TO 'menu_user'@'localhost';
FLUSH PRIVILEGES;
EOF
echo "✅ MySQL 配置完成"

echo "[3/6] 创建应用目录..."
mkdir -p ${APP_DIR}
chown ${USER}:${USER} ${APP_DIR}

echo "[4/6] 克隆项目代码..."
cd ${APP_DIR}
if [ -d ".git" ]; then
    git pull origin main
else
    git clone https://github.com/WEMoon957/meau.git .
fi

echo "[5/6] 创建虚拟环境并安装依赖..."
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[6/6] 初始化数据库和配置..."

# 生成会话 token 签名密钥（HMAC-SHA256）
SESSION_SECRET=$(openssl rand -hex 32)

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
SESSION_SECRET=${SESSION_SECRET}
SESSION_TOKEN_TTL=86400
SESSION_TTL_SECONDS=1800
SESSION_CLEANUP_INTERVAL=300
MAX_SESSIONS=500
MAX_CONCURRENT_CHATS=20
CHAT_RATE_PER_SESSION=30
CHAT_RATE_PER_IP=60
CHAT_RATE_WINDOW=60
SESSION_CREATE_PER_IP=10
SESSION_CREATE_WINDOW=60
EOF

cd ${APP_DIR}/main
source ${APP_DIR}/venv/bin/activate
python init_db.py

cat > /etc/systemd/system/${APP_NAME}.service <<EOF
[Unit]
Description=Menu Recommendation API Service
After=network.target mysql.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=${APP_DIR}/main
Environment="PYTHONPATH=${APP_DIR}"
ExecStart=${APP_DIR}/venv/bin/python api_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${APP_NAME}
systemctl start ${APP_NAME}

cat > /etc/nginx/sites-available/${APP_NAME} <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
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
echo "服务状态:"
systemctl status ${APP_NAME}
echo ""
echo "访问地址: http://你的服务器IP/api/health"
echo ""
echo "后续操作:"
echo "1. 在阿里云安全组放行 80 端口（入方向）"
echo "2. 查看日志: journalctl -u ${APP_NAME} -f"
echo "3. 重启服务: systemctl restart ${APP_NAME}"
echo "4. 配置 HTTPS（可选）: certbot --nginx"
echo ""
echo "⚠️ 注意: SESSION_SECRET 已自动生成并写入 ${APP_DIR}/.env"
echo "   重新运行本脚本会重新生成密钥，导致所有已签发会话 token 失效。"

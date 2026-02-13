+++
date = '2026-02-13T22:08:01+08:00'
draft = false
title = 'OpenClaw 云服务器部署方案'
image = '/images/bg/xinjiang-kanas.jpg'
categories = ['AI', 'Agent']
tags = ['AI', 'OpenClaw']
+++

# OpenClaw 阿里云 ECS（日本）部署方案

## 一、环境信息

| 项目     | 要求                     |
| -------- | ------------------------ |
| 服务器   | 阿里云 ECS（日本）       |
| 操作系统 | Ubuntu 22.04 LTS（推荐） |
| CPU      | ≥ 2 vCPU                 |
| 内存     | ≥ 2 GB（推荐 4 GB）      |
| 存储     | ≥ 40 GB ESSD             |
| Node.js  | v22+                     |
| 必开端口 | 22 (SSH), 80, 18789      |

> [!TIP]
> 日本节点属于海外地域，联网搜索功能不受限制，是很好的选择。

---

## 二、前置准备

### 2.1 获取大模型 API Key

OpenClaw 本身不含大模型，需要对接 API。支持的模型提供商：

- **OpenAI** — `sk-xxx`
- **Google Gemini** — 在 [Google AI Studio](https://aistudio.google.com/apikey) 获取
- **阿里云百炼** — 在阿里云百炼控制台创建
- **Anthropic Claude** — `sk-ant-xxx`

> [!IMPORTANT]
> 请提前准备好至少一个 API Key，部署完成后需要配置。

### 2.2 安全组配置

登录阿里云控制台 → ECS → 安全组，添加以下入方向规则：

| 端口  | 协议 | 来源      | 用途                |
| ----- | ---- | --------- | ------------------- |
| 22    | TCP  | 你的 IP   | SSH 远程连接        |
| 80    | TCP  | 0.0.0.0/0 | HTTP 访问           |
| 443   | TCP  | 0.0.0.0/0 | HTTPS 访问          |
| 18789 | TCP  | 0.0.0.0/0 | OpenClaw Web 控制台 |

---

## 三、服务器初始化

### 3.1 SSH 连接

```bash
ssh root@<你的服务器公网IP>
```

### 3.2 系统更新

```bash
apt update && apt upgrade -y
```

### 3.3 安装基础工具

```bash
apt install -y curl wget git build-essential
```

---

## 四、安装 Node.js 22

```bash
# 使用 NodeSource 安装 Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs

# 验证版本
node -v   # 应显示 v22.x.x
npm -v
```

---

## 五、安装 OpenClaw

### 方式 A：直接安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 安装依赖
npm install

# 复制配置文件
cp .env.example .env
```

编辑配置文件：

```bash
nano .env
```

填入你的 API Key 和其他配置：

```env
# 大模型 API Key（选择你使用的模型填写）
OPENAI_API_KEY=sk-xxx
GOOGLE_API_KEY=your-gemini-api-key
ANTHROPIC_API_KEY=sk-ant-xxx

# 服务端口
PORT=18789

# 访问 Token（用于 Web 控制台登录，自行设定一个强密码）
ACCESS_TOKEN=your-secure-token-here
```

启动服务：

```bash
# 前台运行（测试用）
npm start

# 后台运行（生产环境）
nohup npm start > openclaw.log 2>&1 &
```

### 方式 B：Docker 部署

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# 安装 Docker Compose
apt install -y docker-compose-plugin

# 验证
docker --version
docker compose version
```

创建工作目录：

```bash
mkdir -p /opt/openclaw && cd /opt/openclaw
```

创建 `docker-compose.yml`：

```yaml
version: "3.8"

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: always
    ports:
      - "18789:18789"
      - "80:80"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    environment:
      - TZ=Asia/Tokyo
```

创建 `.env` 文件（同方式 A），然后启动：

```bash
docker compose up -d

# 查看日志
docker compose logs -f
```

---

## 六、配置 Systemd 服务（方式 A 适用）

让 OpenClaw 开机自启：

```bash
cat > /etc/systemd/system/openclaw.service << 'EOF'
[Unit]
Description=OpenClaw AI Assistant
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/openclaw
ExecStart=/usr/bin/node /root/openclaw/index.js
Restart=on-failure
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动
systemctl daemon-reload
systemctl enable openclaw
systemctl start openclaw

# 查看状态
systemctl status openclaw
```

---

## 七、配置 Nginx 反向代理（可选但推荐）

### 7.1 安装 Nginx

```bash
apt install -y nginx
```

### 7.2 配置反向代理

```bash
cat > /etc/nginx/sites-available/openclaw << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名或 IP

    location / {
        proxy_pass http://127.0.0.1:18789;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
EOF

ln -s /etc/nginx/sites-available/openclaw /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

### 7.3 配置 HTTPS（有域名时）

```bash
# 安装 Certbot
apt install -y certbot python3-certbot-nginx

# 申请证书（替换为你的域名）
certbot --nginx -d your-domain.com

# 自动续期
certbot renew --dry-run
```

---

## 八、验证部署

### 8.1 检查服务状态

```bash
# 检查端口是否监听
ss -tlnp | grep 18789

# 检查进程
ps aux | grep openclaw

# 如果用 Docker
docker ps
```

### 8.2 访问 Web 控制台

浏览器打开：

```
http://<你的服务器公网IP>:18789
```

如果配置了 Nginx + 域名：

```
https://your-domain.com
```

输入你在 `.env` 中设定的 `ACCESS_TOKEN` 登录。

### 8.3 功能测试

1. ✅ 登录 Web 控制台
2. ✅ 发送一条消息，确认 AI 能正常回复
3. ✅ 测试联网搜索功能
4. ✅ 测试工具调用

---

## 九、日常运维

### 9.1 常用命令

```bash
# 查看日志（直接安装）
journalctl -u openclaw -f

# 查看日志（Docker）
docker compose logs -f openclaw

# 重启服务
systemctl restart openclaw     # 直接安装
docker compose restart         # Docker

# 更新 OpenClaw
cd /root/openclaw
git pull
npm install
systemctl restart openclaw

# Docker 更新
docker compose pull
docker compose up -d
```

### 9.2 备份

```bash
# 备份数据目录
tar -czf openclaw-backup-$(date +%Y%m%d).tar.gz /root/openclaw/data

# 建议配置定时备份
crontab -e
# 添加：每天凌晨2点备份
# 0 2 * * * tar -czf /backup/openclaw-$(date +\%Y\%m\%d).tar.gz /root/openclaw/data
```

### 9.3 监控

```bash
# 内存使用
free -h

# 磁盘使用
df -h

# Node.js 进程资源
top -p $(pgrep -f openclaw)
```

---

## 十、接入聊天应用

OpenClaw 支持多种主流聊天平台，部署完成后可按需接入。

### 10.1 支持平台一览

| 平台                | 连接方式                    | 适用场景                          |
| ------------------- | --------------------------- | --------------------------------- |
| **WhatsApp**        | QR 码扫描配对（Baileys 库） | 海外个人用，支持一对一和群聊      |
| **Telegram**        | Bot Token（grammY 库）      | 海外个人/团队，配置最简单         |
| **Discord**         | Discord Bot API             | 社区/团队，支持服务器、频道和私信 |
| **微信 / 企业微信** | 阿里云 AppFlow 集成         | 国内用户，支持群聊自然语言交互    |
| **Slack**           | Slack Bot API               | 企业工作场景                      |
| **Signal**          | Signal API                  | 注重隐私的场景                    |
| **iMessage**        | 需 macOS 设备               | Apple 生态用户                    |

### 10.2 Telegram 接入（推荐，最简单）

1. 在 Telegram 中搜索 `@BotFather`，发送 `/newbot` 创建机器人
2. 获取 Bot Token（格式：`123456:ABC-DEF...`）
3. 在 OpenClaw Web 控制台中配置 Telegram Token
4. 在 Telegram 中与你的 Bot 对话即可

### 10.3 WhatsApp 接入

1. 在 OpenClaw Web 控制台中启用 WhatsApp 连接
2. 页面会生成一个 QR 码
3. 打开手机 WhatsApp → 设置 → 已关联设备 → 关联设备 → 扫描 QR 码
4. 扫码成功后即可通过 WhatsApp 与 AI 对话

> [!WARNING]
> WhatsApp 接入使用的是 Baileys（非官方库），建议使用**非主力号**接入，避免被封号风险。

### 10.4 Discord 接入

1. 前往 [Discord Developer Portal](https://discord.com/developers/applications) 创建应用
2. 在 Bot 页面获取 Token，开启 **Message Content Intent**
3. 生成 OAuth2 邀请链接，将 Bot 邀请到你的服务器
4. 在 OpenClaw 中配置 Discord Bot Token

### 10.5 微信 / 企业微信接入

1. 登录阿里云控制台，开通 AppFlow 服务
2. 按引导配置企业微信连接器
3. 将 OpenClaw 的 Webhook 地址填入 AppFlow
4. 在企业微信群聊中 @机器人 即可触发 AI 回复

### 10.6 接入建议

- **个人日常用** → Telegram（零成本，秒配置）
- **国内团队** → 企业微信
- **海外社区** → Discord
- **注重隐私** → Signal

---

## 十一、常见问题

| 问题                | 解决方案                                     |
| ------------------- | -------------------------------------------- |
| 18789 端口无法访问  | 检查安全组规则是否放通                       |
| AI 不回复           | 检查 `.env` 中的 API Key 是否正确            |
| 内存不足 OOM        | 升级到 4GB 内存，或添加 swap                 |
| 搜索功能异常        | 日本节点一般没问题，检查网络连通性           |
| 更新后无法启动      | 重新 `npm install` 安装依赖                  |
| WhatsApp 掉线       | 重新扫描 QR 码，检查网络稳定性               |
| Telegram Bot 无响应 | 检查 Bot Token 是否正确，确认 Bot 未被 Block |

### 添加 Swap（内存不足时）

```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

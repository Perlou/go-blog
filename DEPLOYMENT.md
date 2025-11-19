# 服务器部署指南

本文档提供在阿里云服务器上部署博客的详细步骤。

## 前置要求

### 1. 服务器环境准备

连接到你的阿里云服务器：

```bash
ssh root@your-server-ip
```

#### 安装 Docker

```bash
# 更新软件包
apt update && apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 启动 Docker
systemctl start docker
systemctl enable docker

# 验证安装
docker --version
```

#### 安装 Docker Compose

```bash
# 下载 Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 添加执行权限
chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

### 2. 配置 SSH 密钥（用于 GitHub Actions）

#### 2.1 生成 SSH 密钥（在本地电脑执行）

如果你还没有 SSH 密钥对：

```bash
ssh-keygen -t rsa -b 4096 -C "github-actions" -f ~/.ssh/github_actions_key -N ""
```

这会生成两个文件：

- `~/.ssh/github_actions_key`（私钥）
- `~/.ssh/github_actions_key.pub`（公钥）

#### 2.2 将公钥添加到服务器

```bash
# 复制公钥内容
cat ~/.ssh/github_actions_key.pub

# 登录服务器，添加到 authorized_keys
ssh root@your-server-ip
echo "你的公钥内容" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
exit
```

#### 2.3 测试 SSH 连接

```bash
ssh -i ~/.ssh/github_actions_key root@your-server-ip
```

### 3. 配置 GitHub Secrets

前往你的 GitHub 仓库：`Settings` → `Secrets and variables` → `Actions` → `New repository secret`

添加以下三个 Secrets：

| Name             | Value                | 说明                                        |
| ---------------- | -------------------- | ------------------------------------------- |
| `SERVER_HOST`    | 你的服务器 IP 或域名 | 例如：`123.456.789.0` 或 `perlou.top`       |
| `SERVER_USER`    | SSH 登录用户名       | 通常是 `root`                               |
| `SERVER_SSH_KEY` | SSH 私钥内容         | 复制 `~/.ssh/github_actions_key` 的全部内容 |

获取私钥内容：

```bash
cat ~/.ssh/github_actions_key
```

> **重要**：复制时要包含 `-----BEGIN OPENSSH PRIVATE KEY-----` 和 `-----END OPENSSH PRIVATE KEY-----`

### 4. 域名配置

登录你的域名管理面板（阿里云域名控制台），添加 DNS 记录：

| 类型 | 主机记录 | 记录值        |
| ---- | -------- | ------------- |
| A    | @        | 你的服务器 IP |
| A    | www      | 你的服务器 IP |

等待 DNS 解析生效（通常 10 分钟内）。

验证解析：

```bash
ping perlou.top
```

## 部署流程

### 首次部署

1. **推送代码到 GitHub**

```bash
git add .
git commit -m "feat: add docker and ci/cd configuration"
git push origin main
```

2. **查看 GitHub Actions 执行情况**

前往 GitHub 仓库的 `Actions` 标签页，查看工作流执行状态。

3. **验证部署**

等待 Actions 执行完成后，访问：

- http://perlou.top
- http://www.perlou.top

### 日常更新

每次更新博客内容后：

```bash
git add .
git commit -m "post: add new article"
git push origin main
```

GitHub Actions 会自动触发构建和部署。

## 可选配置：HTTPS/SSL

### 使用 Let's Encrypt 免费证书

1. **安装 Certbot**

```bash
ssh root@your-server-ip
apt install certbot python3-certbot-nginx -y
```

2. **停止当前容器**

```bash
cd /root/go-blog
docker-compose down
```

3. **获取证书**

```bash
certbot certonly --standalone -d perlou.top -d www.perlou.top
```

按提示输入邮箱，证书会保存在 `/etc/letsencrypt/live/perlou.top/`

4. **更新 docker-compose.yml**

编辑 `/root/go-blog/docker-compose.yml`，添加端口和卷挂载：

```yaml
version: "3.8"

services:
  blog:
    image: go-blog:latest
    container_name: perlou-blog
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt:ro
    networks:
      - blog-network

networks:
  blog-network:
    driver: bridge
```

5. **更新 nginx.conf 支持 HTTPS**

需要修改本地的 `nginx.conf`，添加 HTTPS 配置，然后重新推送代码触发部署。

6. **设置自动续期**

```bash
crontab -e
```

添加：

```
0 0 * * * certbot renew --quiet && docker-compose -f /root/go-blog/docker-compose.yml restart
```

## 故障排查

### 查看容器日志

```bash
ssh root@your-server-ip
cd /root/go-blog
docker-compose logs -f
```

### 重新部署

```bash
docker-compose down
docker-compose up -d
```

### 查看容器状态

```bash
docker ps
```

### 测试 Nginx 配置

```bash
docker exec perlou-blog nginx -t
```

## 维护命令

```bash
# 重启容器
docker-compose restart

# 停止容器
docker-compose down

# 查看资源占用
docker stats

# 清理未使用的镜像
docker system prune -a
```

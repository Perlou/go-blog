# 部署指南

本文档提供博客部署的核心步骤。

## 前置要求

- 阿里云服务器（已安装 Docker 和 Docker Compose）
- GitHub 仓库：Perlou/go-blog

## GitHub Secrets 配置

### 1. 生成 SSH 密钥

```bash
ssh-keygen -t rsa -b 4096 -C "github-actions" -f ~/.ssh/github_actions_key -N ""
```

### 2. 添加公钥到服务器

```bash
# 显示公钥
cat ~/.ssh/github_actions_key.pub

# 登录服务器
ssh root@your-server-ip

# 添加公钥
echo "公钥内容" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
exit
```

### 3. 测试连接

```bash
ssh -i ~/.ssh/github_actions_key root@your-server-ip
```

### 4. 配置 GitHub Secrets

访问：`https://github.com/Perlou/go-blog/settings/secrets/actions`

添加三个 Secrets：

| Name             | Value                                           |
| ---------------- | ----------------------------------------------- |
| `SERVER_HOST`    | 服务器 IP 或域名                                |
| `SERVER_USER`    | `root`                                          |
| `SERVER_SSH_KEY` | 私钥完整内容（`cat ~/.ssh/github_actions_key`） |

**重要**：私钥必须包含开头和结尾标记：

```
-----BEGIN OPENSSH PRIVATE KEY-----
... 所有内容 ...
-----END OPENSSH PRIVATE KEY-----
```

## 域名配置

登录域名管理面板，添加 DNS 记录：

| 类型 | 主机记录 | 记录值    |
| ---- | -------- | --------- |
| A    | @        | 服务器 IP |
| A    | www      | 服务器 IP |

等待 DNS 解析生效（10-30 分钟）。

## 部署流程

### 自动部署

```bash
./publish.sh
```

访问 GitHub Actions 查看部署状态：

```
https://github.com/Perlou/go-blog/actions
```

### 验证部署

访问：

- http://perlou.top
- http://www.perlou.top

### 查看日志

```bash
ssh root@your-server-ip
cd /root/go-blog
docker-compose logs -f
```

## 维护命令

```bash
# 重启容器
docker-compose restart

# 停止容器
docker-compose down

# 查看状态
docker ps

# 清理资源
docker system prune -a
```

# GitHub Secrets 快速配置指南

## 问题

GitHub Actions 报错：`Permission denied (publickey,password)`

## 原因

GitHub Secrets 尚未配置或配置不正确。

## 解决步骤

### 1️⃣ 生成 SSH 密钥对（本地执行）

```bash
ssh-keygen -t rsa -b 4096 -C "github-actions" -f ~/.ssh/github_actions_key -N ""
```

### 2️⃣ 将公钥添加到服务器

```bash
# 显示公钥
cat ~/.ssh/github_actions_key.pub

# 登录服务器
ssh root@你的服务器IP

# 在服务器上执行
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 将刚才的公钥内容添加到这个文件
echo "你的公钥内容" >> ~/.ssh/authorized_keys

# 设置权限
chmod 600 ~/.ssh/authorized_keys
exit
```

### 3️⃣ 测试连接

```bash
# 在本地测试
ssh -i ~/.ssh/github_actions_key root@你的服务器IP
```

如果能成功连接，继续下一步。

### 4️⃣ 配置 GitHub Secrets

1. 打开浏览器，访问：`https://github.com/Perlou/go-blog/settings/secrets/actions`

2. 点击 "New repository secret"，添加三个 secrets：

#### Secret 1: SERVER_HOST

```
Name: SERVER_HOST
Value: 你的服务器IP地址（如：123.456.789.0）
```

#### Secret 2: SERVER_USER

```
Name: SERVER_USER
Value: root
```

#### Secret 3: SERVER_SSH_KEY

```
Name: SERVER_SSH_KEY
Value: [粘贴下面命令的完整输出]
```

获取私钥内容：

```bash
cat ~/.ssh/github_actions_key
```

**重要**：确保复制完整内容，包括：

```
-----BEGIN OPENSSH PRIVATE KEY-----
... 所有内容 ...
-----END OPENSSH PRIVATE KEY-----
```

### 5️⃣ 推送更新并测试

```bash
git add .github/workflows/deploy.yml
git commit -m "fix: improve SSH key handling in workflow"
git push
```

### 6️⃣ 查看 GitHub Actions

访问：`https://github.com/Perlou/go-blog/actions`

查看最新的工作流执行情况。

## 常见问题

### Q: 如何知道 Secrets 配置成功？

A: 配置后，在 GitHub Actions 日志中不会再看到 "Permission denied" 错误。

### Q: 私钥格式有什么要求？

A: 必须是完整的私钥内容，包括开头和结尾的标记行，保持原有的换行。

### Q: 可以用已有的 SSH 密钥吗？

A: 可以，但建议单独生成一个专门用于 GitHub Actions 的密钥。

## 验证清单

- [ ] SSH 密钥对已生成
- [ ] 公钥已添加到服务器 `~/.ssh/authorized_keys`
- [ ] 本地能用私钥成功 SSH 连接服务器
- [ ] GitHub Secrets 已配置 `SERVER_HOST`
- [ ] GitHub Secrets 已配置 `SERVER_USER`
- [ ] GitHub Secrets 已配置 `SERVER_SSH_KEY`（完整私钥）
- [ ] 代码已推送到 GitHub
- [ ] GitHub Actions 重新运行

# 改进后的发布工作流

## 🎯 工作流说明

现在的工作流已经改进为：

- **日常开发**：git push 不会触发部署
- **发布上线**：运行 `./publish.sh` 才会触发部署

## 📦 安装 GitHub CLI（可选但推荐）

### macOS

```bash
brew install gh
```

### 首次使用需要登录

```bash
gh auth login
```

选择：

1. GitHub.com
2. HTTPS
3. Yes (login with a web browser)
4. 复制验证码，在浏览器中粘贴

## 🚀 使用方法

### 方式 1：使用 publish.sh（推荐）

```bash
# 1. 编辑文章
code content/post/2025-11-20-new-post.md

# 2. 运行发布脚本
./publish.sh

# 3. 输入提交信息
post: 发布新文章
```

**如果安装了 GitHub CLI**：

- ✅ 自动触发部署

**如果没有安装 GitHub CLI**：

- ⚠️ 需要手动点击链接触发部署

### 方式 2：手动触发

如果你想分开提交代码和部署：

```bash
# 1. 提交代码（不触发部署）
git add .
git commit -m "post: 写了一篇新文章"
git push

# 2. 需要部署时，访问以下链接手动触发
https://github.com/Perlou/go-blog/actions/workflows/deploy.yml
# 点击 "Run workflow" 按钮
```

## 📋 对比

### 之前的工作流

```bash
git push  →  自动部署（无法控制）
```

### 现在的工作流

```bash
# 日常开发
git push  →  只同步代码

# 发布上线
./publish.sh  →  同步代码 + 触发部署
```

## 💡 优势

1. **灵活控制**：可以先提交多个改动，最后一次性部署
2. **节省资源**：避免频繁触发 CI/CD
3. **明确意图**：发布是显式操作，不会意外部署
4. **支持草稿**：可以提交草稿文章但不部署

## 🔍 查看部署状态

访问：https://github.com/Perlou/go-blog/actions

## ⚙️ 技术细节

### GitHub Actions 配置

```yaml
on:
  workflow_dispatch: # 仅手动触发
```

### publish.sh 功能

1. 检查文件改动
2. 提示输入提交信息
3. 提交并推送代码
4. 使用 gh CLI 触发部署（如果已安装）
5. 显示部署链接

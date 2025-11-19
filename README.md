# Perlou's Blog

> slow is fast

这是使用 Hugo 和 Stack 主题构建的个人博客。

## 🚀 快速开始

### 环境要求

- [Hugo Extended](https://gohugo.io/installation/) (v0.152.2+)
- Git

### 本地开发

克隆项目：

```bash
git clone <repository-url>
cd go-blog
```

初始化主题（如果是首次克隆）：

```bash
git submodule update --init --recursive
```

启动开发服务器：

```bash
hugo server
```

访问 http://localhost:1313 即可预览博客。

Hugo 服务器会自动监听文件变化并实时刷新页面。

## 📝 写作指南

### 创建新文章

```bash
hugo new content/post/YYYY-MM-DD-article-title.md
```

### 文章格式

文章使用 Front Matter (TOML 格式) 配置元数据：

```toml
+++
date = '2025-11-19T13:45:49+08:00'
draft = true
title = '文章标题'
categories = ['分类']
tags = ['标签1', '标签2']
+++

文章内容使用 Markdown 格式...
```

### 发布文章

将文章的 `draft` 从 `true` 改为 `false`，或者直接删除该行。

## � 发布工作流

本项目使用**手动触发**的 CI/CD 流程，让你完全掌控何时部署。

### 快速发布

```bash
./publish.sh
```

这个脚本会自动：

1. 📋 检查并显示所有改动
2. 💬 提示输入提交信息
3. 📦 提交并推送代码到 GitHub
4. 🚀 自动触发部署到生产环境

### 日常开发 vs 发布上线

**日常开发**（不触发部署）：

```bash
git add .
git commit -m "draft: 正在写文章..."
git push
```

**发布上线**（触发部署）：

```bash
./publish.sh
# 输入：post: 发布新文章《标题》
```

### 前置要求

需要安装 GitHub CLI 以实现自动触发：

```bash
# 安装
brew install gh

# 登录
gh auth login
```

详细说明请参考：[PUBLISH_WORKFLOW.md](PUBLISH_WORKFLOW.md)

## �📁 项目结构

```
.
├── archetypes/       # 内容模板
├── assets/           # 静态资源（图片、CSS、JS等）
├── content/          # 博客内容
│   ├── page/        # 页面（关于、归档、搜索等）
│   └── post/        # 博客文章
├── layouts/          # 自定义布局模板
├── static/           # 静态文件（不会被处理）
├── themes/           # 主题目录
│   └── hugo-theme-stack/
├── hugo.yaml         # Hugo 配置文件
└── README.md         # 项目说明
```

## ⚙️ 配置说明

主要配置文件为 `hugo.yaml`，包含以下配置：

- **站点信息**：标题、副标题、头像等
- **语言设置**：默认为简体中文（zh-cn）
- **菜单配置**：导航栏、社交链接等
- **侧边栏组件**：搜索、归档、分类、标签云等
- **文章设置**：目录、阅读时间、许可协议等

## 🎨 主题

本博客使用 [Hugo Theme Stack](https://github.com/CaiJimmy/hugo-theme-stack) 主题。

- 简洁优雅的设计
- 响应式布局
- 支持深色模式
- 内置搜索功能
- 丰富的组件和小工具

## 📦 部署

### 自动化部署（推荐）

本项目已配置完整的 **Docker + GitHub Actions** 自动化部署流程。

#### 部署流程

```
./publish.sh → GitHub Actions → Docker 构建 → 部署到服务器 → 自动上线
```

**特点**：

- ✅ 手动触发，完全可控
- ✅ 自动构建 Docker 镜像
- ✅ 自动传输并部署到服务器
- ✅ 零停机更新

**详细部署指南**：

- [PUBLISH_WORKFLOW.md](PUBLISH_WORKFLOW.md) - 发布工作流说明
- [DEPLOYMENT.md](DEPLOYMENT.md) - 服务器部署指南
- [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md) - GitHub Secrets 配置

### 手动部署

如果需要手动部署：

```bash
# 构建 Docker 镜像
docker build -t go-blog:latest .

# 启动容器
docker-compose up -d
```

### 构建静态文件

```bash
hugo --minify
```

生成的静态文件位于 `public/` 目录。

- **GitHub Pages**：将 `public/` 目录推送到 GitHub Pages
- **Netlify**：连接 Git 仓库自动部署
- **Vercel**：支持 Hugo 项目的一键部署
- **自托管**：将 `public/` 目录中的文件上传到任何 Web 服务器

## 🔗 链接

- 博客作者：[@Perlou](https://github.com/Perlou)
- Twitter：[@perlou666](https://x.com/perlou666)

## 📄 许可

文章内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可协议。

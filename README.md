# Perlou's Blog

> slow is fast

基于 Hugo + Stack 主题构建的现代化个人技术博客，专注于前端开发、全栈技术和 AI 应用。

🌐 **在线访问**：[perlou.top](https://perlou.top)

## 🚀 快速开始

### 环境要求

- [Hugo Extended](https://gohugo.io/installation/) v0.152.2+
- Git
- (可选) Docker - 用于容器化部署

### 本地开发

```bash
# 克隆项目
git clone https://github.com/Perlou/go-blog.git
cd go-blog

# 初始化主题子模块
git submodule update --init --recursive

# 启动开发服务器
hugo server
```

访问 http://localhost:1313 即可预览博客。

Hugo 会自动监听文件变化并实时刷新页面。

## 📝 写作指南

### 创建新文章

```bash
# 按照日期命名规范创建文章
hugo new content/post/YYYY-MM-DD-article-title.md

# 例如
hugo new content/post/2025-11-20-hugo-blog-guide.md
```

### 文章 Front Matter

```toml
+++
date = '2025-11-20T10:00:00+08:00'
draft = false
title = '文章标题'
image = '/images/covers/article-cover.jpg'  # 封面图（可选）
categories = ['技术']
tags = ['Hugo', 'Blog', '教程']
+++

文章内容使用 Markdown 格式...
```

### 图片管理

将图片放在 `static/images/` 目录：

```markdown
![图片描述](/images/your-image.jpg)
```

**图片优化建议**：

- 封面图：建议尺寸 1200x630px
- 文章配图：宽度不超过 1440px
- 使用 WebP 格式可进一步减小体积

### 发布文章

1. 将 `draft: false` 或删除 `draft` 字段
2. 运行 `./publish.sh` 自动发布

## 🔄 发布工作流

本项目采用**手动触发**的 CI/CD 流程，完全可控。

### 一键发布

```bash
./publish.sh
```

脚本会自动：

1. 📋 检查并显示所有改动
2. 💬 提示输入提交信息
3. 📦 提交并推送到 GitHub
4. 🚀 触发 GitHub Actions 自动部署

### 日常开发 vs 发布上线

**日常开发**（不触发部署）：

```bash
git add .
git commit -m "draft: 正在写作..."
git push
```

**发布上线**（触发自动部署）：

```bash
./publish.sh
# 输入提交信息，如：post: 发布新文章《Hugo 博客搭建指南》
```

### 前置要求

需要安装 GitHub CLI：

```bash
# macOS
brew install gh

# 登录
gh auth login
```

详细说明：[PUBLISH_WORKFLOW.md](PUBLISH_WORKFLOW.md)

## 📁 项目结构

```
.
├── .github/
│   └── workflows/          # GitHub Actions 工作流
├── archetypes/             # 内容模板
├── content/
│   ├── page/              # 页面（关于、归档、搜索、链接）
│   └── post/              # 博客文章
├── layouts/               # 自定义布局模板
│   ├── partials/          # 部分模板
│   └── shortcodes/        # 短代码
├── scripts/               # 工具脚本
│   └── test-cdn.sh       # CDN 验证脚本
├── static/
│   ├── images/           # 图片资源
│   ├── favicon.ico       # 网站图标
│   └── robots.txt        # 搜索引擎爬虫配置
├── themes/
│   └── hugo-theme-stack/ # Stack 主题（submodule）
├── Dockerfile            # Docker 镜像构建
├── docker-compose.yml    # Docker Compose 配置
├── hugo.yaml             # Hugo 主配置文件
├── nginx.conf            # Nginx 服务器配置
├── publish.sh            # 一键发布脚本
└── README.md
```

## ⚙️ 配置说明

### Hugo 配置（`hugo.yaml`）

主要配置项：

```yaml
baseurl: https://perlou.top/
languageCode: zh-cn
title: Perlou

params:
  # SEO 优化
  description: "前端工程师 Perlou 的技术博客..."
  keywords: [前端开发, Vue, React, Go, 全栈开发]

  # 图片响应式处理
  imageProcessing:
    cover:
      enabled: true
      sizes: [480, 720, 1024, 1440]

  # Giscus 评论系统
  comments:
    enabled: true
    provider: giscus
```

### Nginx 配置（`nginx.conf`）

- Cloudflare 真实 IP 检测
- 静态资源长期缓存（1 年）
- Gzip + Brotli 压缩
- 安全响应头

### Cloudflare CDN

配置指南：[CLOUDFLARE_DASHBOARD_GUIDE.md](CLOUDFLARE_DASHBOARD_GUIDE.md)

## 🎨 主题

使用 [Hugo Theme Stack](https://github.com/CaiJimmy/hugo-theme-stack)，具有以下特点：

- ✨ 现代简洁的设计
- 📱 完美的移动端体验
- 🌓 深色/浅色模式切换
- 🔍 内置搜索功能
- 📊 丰富的侧边栏小工具
- 💬 评论系统集成
- 📈 阅读进度指示器

## 📦 部署

### 自动化部署（推荐）

本项目使用 **Docker + GitHub Actions** 实现全自动化部署。

#### 部署流程

```
代码提交 → GitHub Actions → Docker 构建 → SSH 传输 → 服务器部署 → 自动上线
```

**特性**：

- ✅ 手动触发，安全可控
- ✅ 自动构建优化的 Docker 镜像
- ✅ 零停机部署（先启动新容器，再停止旧容器）
- ✅ 健康检查和自动回滚

#### 配置指南

- [DEPLOYMENT.md](DEPLOYMENT.md) - 完整部署指南
- [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md) - GitHub Secrets 配置
- [PUBLISH_WORKFLOW.md](PUBLISH_WORKFLOW.md) - 发布工作流说明

### Docker 部署

```bash
# 构建镜像
docker build -t go-blog:latest .

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 手动构建

```bash
# 构建静态文件
hugo --minify

# 生成的文件在 public/ 目录
```

## 🚀 性能优化

本博客采用**四层优化策略**，确保极致的访问速度：

### 1. 构建时优化（Hugo）

- HTML/CSS/JS 压缩（minify）
- 响应式图片生成（4 种尺寸）
- 资源指纹（fingerprinting）

### 2. 服务器优化（Nginx）

- Gzip 压缩（级别 6）
- 静态资源长期缓存（1 年）
- 安全响应头

### 3. CDN 优化（Cloudflare）

- 全球 200+ 节点分发
- Brotli 智能压缩
- 边缘缓存（Edge Cache）
- HTTP/3 + QUIC 协议

### 4. 浏览器优化

- Service Worker 离线缓存
- 预加载关键资源
- 懒加载图片

**性能指标**：

- 🇨🇳 国内访问：0.3-0.8s（提升 80%+）
- 🌏 海外访问：0.5-1s（提升 90%+）
- 📦 资源体积：减少 20-30%
- 🎯 缓存命中率：80%+

## 📚 文档

- [CLOUDFLARE_CDN_SETUP.md](CLOUDFLARE_CDN_SETUP.md) - CDN 加速配置概览
- [CLOUDFLARE_DASHBOARD_GUIDE.md](CLOUDFLARE_DASHBOARD_GUIDE.md) - Cloudflare 详细配置
- [GISCUS_SETUP_GUIDE.md](GISCUS_SETUP_GUIDE.md) - Giscus 评论系统配置
- [ANALYTICS_SETUP.md](ANALYTICS_SETUP.md) - Google Analytics 配置
- [IMAGE_OPTIMIZATION.md](IMAGE_OPTIMIZATION.md) - 图片优化指南
- [DEPLOYMENT.md](DEPLOYMENT.md) - 服务器部署指南

## 🛠️ 技术栈

**核心**：

- [Hugo](https://gohugo.io/) - 静态网站生成器
- [Stack Theme](https://github.com/CaiJimmy/hugo-theme-stack) - Hugo 主题

**部署**：

- [Docker](https://www.docker.com/) - 容器化
- [Nginx](https://nginx.org/) - Web 服务器
- [GitHub Actions](https://github.com/features/actions) - CI/CD

**服务**：

- [Cloudflare](https://www.cloudflare.com/) - CDN + SSL
- [Giscus](https://giscus.app/) - 评论系统
- [Google Analytics](https://analytics.google.com/) - 访问统计

## 🔧 开发工具

### 验证 CDN 状态

```bash
./scripts/test-cdn.sh
```

### 本地预览生产构建

```bash
hugo --minify
hugo server --source public
```

### 检查断开的链接

```bash
hugo server
# 在另一个终端
wget --spider -r -nd -nv http://localhost:1313
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如果这个项目对你有帮助，欢迎 Star ⭐

## 🔗 链接

- 📝 **博客**：[perlou.top](https://perlou.top)
- 🐙 **GitHub**：[@Perlou](https://github.com/Perlou)
- 🐦 **Twitter**：[@perlou666](https://x.com/perlou666)

## 📄 许可

- **代码**：MIT License
- **文章内容**：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

<p align="center">
  Made with ❤️ by Perlou
</p>

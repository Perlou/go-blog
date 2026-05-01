# Perlou's Blog

> slow is fast

基于 Hugo + Stack 主题构建的现代化个人技术博客，主要分享 AI、全栈开发与工具实践相关的内容。

🌐 **在线访问**：[perlou.top](https://perlou.top)

---

## 🚀 快速开始

### 环境要求

- [Hugo Extended](https://gohugo.io/installation/) v0.152.2+
- Git
- 可选：[webp](https://formulae.brew.sh/formula/webp) + [jq](https://stedolan.github.io/jq/) —— 用于封面图生成与转码

### 本地开发

```bash
git clone https://github.com/Perlou/go-blog.git
cd go-blog
git submodule update --init --recursive    # 拉取 Stack 主题
hugo server                                  # 访问 http://localhost:1313
```

---

## 📝 写作工作流

### 创建文章

```bash
hugo new content/post/YYYY-MM-DD-article-title.md
```

### Front Matter 规范

```toml
+++
date = '2026-05-01T10:00:00+08:00'
draft = false
title = '文章标题'
image = '/images/bg/your-image.webp'
categories = ['AI', '技术']
tags = ['AI', 'Agents', 'LLM']
+++
```

### 封面图：本地生成 / 已有 / AI 生成

封面图存放在 `assets/images/bg/` 下（WebP 格式），front matter 中写
`/images/bg/<文件名>.webp`。Hugo 会自动生成 800w / 1600w 响应式缩略图。

**已有图片**：从 `assets/images/bg/` 目录里挑一张（30+ 张精选风景与技术主题图）。

**AI 生成新封面**（推荐，[Nano Banana Pro](https://blog.google/innovation-and-ai/products/nano-banana-pro/)）：

```bash
# 一次性配置
brew install webp jq
cp .env.example .env                # 编辑 .env 填入 GEMINI_API_KEY

# 随机生成一张（最常用）
./scripts/generate-bg-batch.sh -n 1

# 随机生成 5 张
./scripts/generate-bg-batch.sh -n 5

# 跑完 manifest 全部还没生成的（manifest 共 65 条，覆盖 38 个国家 / 地区）
./scripts/generate-bg-batch.sh

# 指定主题单张生成
./scripts/generate-bg.sh tibet-yamdrok "Yamdrok sacred lake at sunrise..."
```

详见 [`scripts/README.md`](./scripts/README.md) 与 [`docs/IMAGE_WEBP_GUIDE.md`](./docs/IMAGE_WEBP_GUIDE.md)。

### 发布

```bash
./publish.sh
# 输入 commit message（如 `post: 发布新文章《标题》`）
# 自动 commit + push + 触发 GitHub Actions 部署
```

详见 [`docs/PUBLISH_WORKFLOW.md`](./docs/PUBLISH_WORKFLOW.md)。

---

## 📦 部署

使用 Docker + GitHub Actions 自动化部署到阿里云服务器，前置 Cloudflare CDN。

### 本地 Docker 构建

```bash
docker build -t go-blog:latest .
docker-compose up -d
```

完整部署流程：[`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md)。

---

## 🏗️ 项目结构

```
go-blog/
├─ content/
│  ├─ post/                    博客文章（按日期命名）
│  └─ page/                    单页（about / archives / links / search）
├─ assets/
│  └─ images/bg/               封面图（WebP）
├─ static/                     静态文件（favicon、图标等）
├─ layouts/                    主题 override（不改主题源码！）
│  ├─ partials/
│  │  ├─ helper/image.html       响应式图片处理（800w / 1600w）
│  │  ├─ head/custom.html        DNS prefetch + 头像预加载
│  │  ├─ head/style.html         CSS preload
│  │  └─ comments/provider/giscus.html  评论懒加载
│  ├─ index.html / index.json    首页 + 搜索索引
│  └─ 404.html
├─ themes/hugo-theme-stack/    主题（git submodule，禁修改）
├─ scripts/
│  ├─ generate-bg.sh             AI 生成单张封面图
│  ├─ generate-bg-batch.sh       批量 / 随机生成
│  ├─ landscape-prompts.json     65 条精选风景图 manifest
│  └─ test-cdn.sh                CDN 测试
├─ skills/                     可复用的 Claude Code skill 工件
├─ docs/                       部署 / Cloudflare / Giscus / 图片 / 发布指南
├─ .agent/                     给 AI 协作者的写作模板与项目规则
├─ hugo.yaml                   主配置（站点 / SEO / 评论 / 分析）
├─ Dockerfile                  多阶段构建（Hugo build + Nginx serve）
├─ nginx.conf                  Gzip + stale-while-revalidate 缓存
├─ publish.sh                  一键发布脚本
└─ CLAUDE.md                   给 AI 协作者的工作约束
```

---

## ⚡ 性能优化

通过 `layouts/partials/` 下的 layout override 实现，**不修改主题源码**：

| 优化项 | 文件 | 说明 |
|---|---|---|
| 图片处理管道 | `helper/image.html` | 封面图自动生成 800w / 1600w srcset |
| 资源预连接 | `head/custom.html` | dns-prefetch + preconnect (Giscus / GA) |
| CSS 预加载 | `head/style.html` | 主样式 preload 加速 FCP |
| 头像预加载 | `head/custom.html` | 首页 LCP 优化 |
| 评论懒加载 | `comments/provider/giscus.html` | IntersectionObserver 延迟加载 |
| Nginx 静态缓存 | `nginx.conf` | Gzip + stale-while-revalidate |
| Cloudflare CDN | — | 全球加速 + SSL |

---

## 🤖 AI 协作

本项目对 AI 协作友好。以下文件帮助 AI 助手快速上手：

| 文件 | 内容 |
|---|---|
| [`CLAUDE.md`](./CLAUDE.md) | 给 AI 协作者的工作约束（红线 / 命令 / 阅读顺序） |
| [`.agent/rules.md`](./.agent/rules.md) | 项目规则与编码规范 |
| [`.agent/blog-writing-prompt-template.md`](./.agent/blog-writing-prompt-template.md) | 文章写作提示词模板 |
| [`skills/`](./skills/) | 可被 Claude Code 加载的 skill 包（写作 / 配图 / 主题改造） |

---

## 🎨 主题

[Hugo Theme Stack](https://github.com/CaiJimmy/hugo-theme-stack) —— 现代简洁、响应式、深色模式。
通过 git submodule 引入；**所有自定义都通过 `layouts/` override**，不改 `themes/` 源码。

---

## 📚 文档

| 文档 | 内容 |
|---|---|
| [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) | 服务器部署指南 |
| [`docs/CLOUDFLARE_DASHBOARD_GUIDE.md`](./docs/CLOUDFLARE_DASHBOARD_GUIDE.md) | Cloudflare 配置 |
| [`docs/GISCUS_SETUP_GUIDE.md`](./docs/GISCUS_SETUP_GUIDE.md) | Giscus 评论系统 |
| [`docs/IMAGE_WEBP_GUIDE.md`](./docs/IMAGE_WEBP_GUIDE.md) | WebP 转换指南 |
| [`docs/PUBLISH_WORKFLOW.md`](./docs/PUBLISH_WORKFLOW.md) | 发布工作流 |
| [`scripts/README.md`](./scripts/README.md) | 封面图 AI 生成脚本说明 |

---

## 🛠️ 技术栈

- [Hugo](https://gohugo.io/) Extended v0.152.2+ —— 静态站点生成器
- [Stack Theme](https://github.com/CaiJimmy/hugo-theme-stack) —— Hugo 主题
- [Docker](https://www.docker.com/) + [Nginx](https://nginx.org/) —— 容器化部署
- [GitHub Actions](https://github.com/features/actions) —— CI/CD
- [Cloudflare](https://www.cloudflare.com/) —— CDN + SSL
- [Giscus](https://giscus.app/) —— 评论系统
- [Google AI Studio](https://aistudio.google.com/) (Nano Banana Pro) —— 封面图生成

---

## 🔗 链接

- 📝 **博客**：[perlou.top](https://perlou.top)
- 🐙 **GitHub**：[@Perlou](https://github.com/Perlou)
- 🐦 **Twitter**：[@perlou666](https://x.com/perlou666)

---

## 📄 许可

- **代码**：MIT License
- **文章内容**：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

<p align="center">
  Made with ❤️ by Perlou
</p>

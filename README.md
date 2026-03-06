# Perlou's Blog

> slow is fast

基于 Hugo + Stack 主题构建的现代化个人技术博客。

🌐 **在线访问**：[perlou.top](https://perlou.top)

## 🚀 快速开始

### 环境要求

- [Hugo Extended](https://gohugo.io/installation/) v0.152.2+
- Git

### 本地开发

```bash
# 克隆项目
git clone https://github.com/Perlou/go-blog.git
cd go-blog

# 初始化主题
git submodule update --init --recursive

# 启动开发服务器
hugo server
```

访问 http://localhost:1313 预览博客。

## 📝 写作指南

### 创建文章

```bash
hugo new content/post/YYYY-MM-DD-article-title.md
```

### Front Matter 示例

```toml
+++
date = '2025-11-20T10:00:00+08:00'
draft = false
title = '文章标题'
image = '/images/bg/your-image.webp'
categories = ['技术']
tags = ['Hugo', 'Blog']
+++
```

### 封面图片

封面图放在 `assets/images/bg/` 目录下（WebP 格式），front matter 中写 `/images/bg/文件名.webp`。Hugo 会自动生成 800w/1600w 响应式缩略图。

详见：[IMAGE_WEBP_GUIDE.md](./docs/IMAGE_WEBP_GUIDE.md)

### 发布

```bash
./publish.sh
```

详见：[PUBLISH_WORKFLOW.md](./docs/PUBLISH_WORKFLOW.md)

## 📦 部署

使用 Docker + GitHub Actions 自动化部署。

### Docker 快速部署

```bash
docker build -t go-blog:latest .
docker-compose up -d
```

### 手动构建

```bash
hugo --minify
```

配置指南：[DEPLOYMENT.md](./docs/DEPLOYMENT.md)

## ⚙️ 核心配置

- **Hugo 配置**：`hugo.yaml` - 站点设置、SEO、评论系统
- **Nginx**：静态资源缓存、Gzip 压缩、stale-while-revalidate
- **CDN**：Cloudflare 全球加速 - [配置指南](./docs/CLOUDFLARE_DASHBOARD_GUIDE.md)

## ⚡ 性能优化

项目通过 `layouts/partials/` 下的 layout override 实现性能优化（不修改主题文件，升级主题无影响）：

| 优化项       | 文件                            | 说明                                   |
| ------------ | ------------------------------- | -------------------------------------- |
| 图片处理管道 | `helper/image.html`             | 封面图自动生成 800w/1600w srcset       |
| 资源预连接   | `head/custom.html`              | dns-prefetch + preconnect (Giscus, GA) |
| CSS 预加载   | `head/style.html`               | 主样式 preload 加速 FCP                |
| 头像预加载   | `head/custom.html`              | 首页 LCP 优化                          |
| 评论懒加载   | `comments/provider/giscus.html` | IntersectionObserver 延迟加载          |

图片统一存放在 `assets/images/bg/`（WebP 格式），Hugo 构建时自动裁切生成响应式尺寸。

## 🎨 主题

[Hugo Theme Stack](https://github.com/CaiJimmy/hugo-theme-stack) - 现代简洁、响应式设计、支持深色模式

## 📚 文档

- [DEPLOYMENT.md](./docs/DEPLOYMENT.md) - 服务器部署指南
- [CLOUDFLARE_DASHBOARD_GUIDE.md](./docs/CLOUDFLARE_DASHBOARD_GUIDE.md) - Cloudflare 配置
- [GISCUS_SETUP_GUIDE.md](./docs/GISCUS_SETUP_GUIDE.md) - 评论系统配置
- [PUBLISH_WORKFLOW.md](./docs/PUBLISH_WORKFLOW.md) - 发布工作流

## 🛠️ 技术栈

- [Hugo](https://gohugo.io/) - 静态网站生成器
- [Stack Theme](https://github.com/CaiJimmy/hugo-theme-stack) - Hugo 主题
- [Docker](https://www.docker.com/) + [Nginx](https://nginx.org/) - 容器化部署
- [GitHub Actions](https://github.com/features/actions) - CI/CD
- [Cloudflare](https://www.cloudflare.com/) - CDN + SSL
- [Giscus](https://giscus.app/) - 评论系统

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

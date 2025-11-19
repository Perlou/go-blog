# Perlou's Blog 项目规则

## 项目概述

这是一个基于 Hugo 静态网站生成器构建的个人技术博客项目，使用 Hugo Theme Stack 主题，主要分享前端、全栈开发和 AI 应用相关的技术内容。

- **博客地址**: https://perlou.top
- **仓库**: Perlou/go-blog
- **作者**: Perlou ([@Perlou](https://github.com/Perlou))
- **语言**: 简体中文 (zh-cn)
- **座右铭**: slow is fast

## 技术栈

### 核心技术

- **静态站点生成器**: [Hugo](https://gohugo.io/) (v0.152.2+, Extended 版本)
- **主题**: [Hugo Theme Stack](https://github.com/CaiJimmy/hugo-theme-stack) (通过 Git Submodule 管理)
- **配置格式**: YAML (`hugo.yaml`)
- **内容格式**: Markdown + TOML Front Matter

### 部署与运维

- **容器化**: Docker + Docker Compose
- **Web 服务器**: Nginx (Alpine)
- **CI/CD**: GitHub Actions (手动触发)
- **部署方式**: 自动化部署到阿里云服务器
- **SSL/HTTPS**: 支持 (可选配置 Let's Encrypt)

### 功能特性

- **评论系统**: Giscus (基于 GitHub Discussions)
- **分析工具**: Google Analytics (G-CTWSKJMNN4)
- **搜索功能**: 内置 JSON 索引搜索
- **图片优化**: 响应式图片处理 (多尺寸: 480px, 720px, 1024px, 1440px)
- **主题模式**: 支持自动/浅色/深色模式切换
- **SEO 优化**: 完整的 meta 标签、Open Graph、Twitter Cards

## 项目结构

\`\`\`
/Users/perlou/Desktop/personal/go-blog/
├── .github/ # GitHub Actions 工作流
├── archetypes/ # 内容模板（default.md）
├── assets/ # 静态资源（图片、CSS、JS）
├── content/ # 博客内容
│ ├── page/ # 页面（关于、归档、搜索、友链）
│ └── post/ # 博客文章（按日期命名）
├── data/ # 数据文件
├── i18n/ # 国际化翻译
├── layouts/ # 自定义布局模板
│ ├── 404.html # 自定义 404 页面
│ ├── \_default/ # 默认布局
│ └── partials/ # 部分模板（header、footer、article-list 等）
├── static/ # 静态文件（favicon.ico、robots.txt、images/）
├── themes/ # 主题目录
│ └── hugo-theme-stack/
├── public/ # 构建输出（不提交到 Git）
├── resources/ # Hugo 资源缓存
├── hugo.yaml # Hugo 主配置文件
├── Dockerfile # Docker 镜像构建
├── docker-compose.yml # Docker Compose 配置
├── nginx.conf # Nginx 配置
├── publish.sh # 发布脚本
├── README.md # 项目说明
├── DEPLOYMENT.md # 部署指南
├── PUBLISH_WORKFLOW.md # 发布工作流
├── GISCUS_SETUP_GUIDE.md # Giscus 配置指南
├── GITHUB_SECRETS_SETUP.md # GitHub Secrets 配置
├── ANALYTICS_SETUP.md # 分析工具配置
├── IMAGE_OPTIMIZATION.md # 图片优化说明
└── CLOUDFLARE_CDN_SETUP.md # Cloudflare CDN 配置
\`\`\`

## 编码规范与最佳实践

### 文章创建

1. **文件命名规范**: 使用 `YYYY-MM-DD-article-title.md` 格式

   ```bash
   # 推荐
   content/post/2025-11-19-random-thoughts.md

   # 不推荐
   content/post/random-thoughts.md
   content/post/2025-random.md
   ```

2. **创建新文章**:

   ```bash
   hugo new content/post/YYYY-MM-DD-article-title.md
   ```

3. **Front Matter 格式**:
   ```toml
   +++
   date = '2025-11-19T13:45:49+08:00'
   draft = false              # false 表示发布，true 表示草稿
   title = '文章标题'
   categories = ['分类']       # 单个或多个分类
   tags = ['标签1', '标签2']   # 相关标签
   image = 'cover.jpg'        # 可选：封面图片
   +++
   ```

### 配置文件管理

1. **主配置文件**: `hugo.yaml`

   - 站点基本信息（baseurl, title, languageCode）
   - SEO 配置（description, keywords, author）
   - 菜单配置（main menu, social links）
   - 侧边栏组件（widgets）
   - 评论系统、分析工具等

2. **主题配置**: `themes/hugo-theme-stack/hugo.yaml`

   - 仅包含主题默认配置
   - **不要直接修改此文件**，所有自定义配置应在根目录 `hugo.yaml` 中覆盖

3. **环境特定配置**:
   - 本地开发: `baseurl: http://localhost:1313/`
   - 生产环境: `baseurl: https://perlou.top/`
   - 使用 Dockerfile 构建时自动指定 baseURL

### 静态资源管理

1. **图片存放位置**:

   - 文章内图片: `static/images/posts/YYYY-MM-DD-article-title/`
   - 通用图片: `static/images/`
   - 头像: 使用 GitHub Avatar (`https://avatars.githubusercontent.com/u/12897436?v=4`)

2. **图片优化**:

   - 启用响应式图片处理
   - 封面图片质量: 85%
   - 内容图片质量: 80%
   - 自动生成多尺寸版本

3. **Favicon**:
   - 位置: `static/favicon.ico`
   - 配置: `favicon: favicon.ico?v=2` (添加版本号以强制刷新缓存)

### 自定义布局

1. **布局优先级**:

   ```
   layouts/ (项目自定义) > themes/hugo-theme-stack/layouts/ (主题默认)
   ```

2. **常见自定义**:

   - `layouts/404.html`: 自定义 404 页面
   - `layouts/partials/article-list/`: 文章列表布局
   - `layouts/partials/head/`: 自定义 head 标签

3. **修改建议**:
   - 先复制主题中的原始文件到 `layouts/` 对应位置
   - 在副本上进行修改
   - 保留清晰的注释说明修改原因

### SEO 最佳实践

1. **必须配置的 SEO 字段**:

   - `title`: 页面标题
   - `description`: 站点描述
   - `keywords`: 关键词列表
   - `author.name` & `author.email`
   - `images`: 用于 Open Graph 的默认图片

2. **文章 SEO**:

   - 每篇文章必须有标题和日期
   - 使用描述性的分类和标签
   - 可选：为重要文章添加自定义 `image`

3. **URL 结构**:
   - 文章: `/p/:slug/` (slug 基于文章标题)
   - 页面: `/:slug/`

## 工作流程

### 本地开发

1. **启动开发服务器**:

   ```bash
   hugo server
   # 访问 http://localhost:1313
   ```

2. **实时预览**:

   - Hugo 服务器自动监听文件变化
   - 浏览器自动刷新

3. **查看草稿**:
   ```bash
   hugo server -D
   ```

### 发布流程

#### 推荐方式：使用 publish.sh

```bash
# 1. 编辑内容
vim content/post/2025-11-20-new-post.md

# 2. 运行发布脚本
./publish.sh

# 3. 输入提交信息
# 格式: <type>: <description>
# 例如: post: 发布新文章《标题》
```

**publish.sh 功能**:

1. 检查并显示所有文件改动
2. 提示输入提交信息
3. 提交并推送代码到 GitHub
4. 自动触发 GitHub Actions 部署（需要安装 `gh` CLI）

#### 提交信息规范

- `post: 发布新文章《标题》`
- `draft: 正在写文章...`
- `feat: 添加新功能`
- `fix: 修复 Bug`
- `style: 样式调整`
- `docs: 更新文档`
- `config: 配置修改`

### 部署流程

#### 自动化部署（推荐）

1. **触发方式**:

   - 运行 `./publish.sh` (自动触发)
   - 或访问 GitHub Actions 页面手动触发

2. **部署流程**:

   ```
   ./publish.sh → GitHub Actions → Docker 构建 → 部署到服务器 → 自动上线
   ```

3. **查看状态**:
   - GitHub Actions: https://github.com/Perlou/go-blog/actions
   - 容器日志: `ssh root@server && docker-compose logs -f`

#### 手动部署

```bash
# 构建静态文件
hugo --minify

# 或使用 Docker
docker build -t go-blog:latest .
docker-compose up -d
```

### 维护命令

```bash
# 更新主题
git submodule update --remote --merge

# 清理缓存
hugo --cleanDestinationDir

# 重启容器
docker-compose restart

# 查看容器日志
docker-compose logs -f

# 测试 Nginx 配置
docker exec perlou-blog nginx -t

# 清理未使用的 Docker 资源
docker system prune -a
```

## AI 助手工作规范

### 理解项目上下文

1. **关键文件**:

   - `hugo.yaml`: 了解站点配置和功能
   - `README.md`: 了解项目整体结构
   - `content/post/`: 查看现有文章的风格和格式
   - 部署相关文档: 了解工作流程

2. **依赖关系**:
   - 主题作为 Git Submodule，不要直接修改
   - 自定义配置应在根目录 `hugo.yaml` 中进行
   - 布局自定义应在项目 `layouts/` 目录中进行

### 执行任务

1. **创建内容**:

   - 使用 `hugo new` 命令创建文章
   - 遵循文件命名规范
   - 正确设置 Front Matter

2. **修改配置**:

   - 备份原配置（建议添加到 Git）
   - 仅修改必要的配置项
   - 验证 YAML 语法正确性

3. **自定义布局**:

   - 先查看主题原始布局
   - 复制到项目 `layouts/` 后再修改
   - 添加注释说明修改理由

4. **部署相关**:
   - 熟悉 `publish.sh` 脚本的使用
   - 了解 GitHub Secrets 配置
   - 知道如何查看部署日志

### 验证与测试

1. **本地验证**:

   ```bash
   # 清理并重新启动
   hugo --cleanDestinationDir
   hugo server
   ```

2. **检查清单**:

   - [ ] 文章/页面正常显示
   - [ ] 图片正确加载
   - [ ] 链接可以点击
   - [ ] 分类/标签正确
   - [ ] SEO 信息完整
   - [ ] 响应式布局正常

3. **部署前检查**:
   - [ ] 所有 draft 文章已设置为 false（如需发布）
   - [ ] 图片路径使用相对路径或 CDN
   - [ ] 外部链接使用 HTTPS
   - [ ] 配置文件 baseurl 正确

## 故障排查

### 常见问题

1. **图片不显示**:

   - 检查图片路径是否正确
   - 确认图片在 `static/` 目录中
   - 验证文件名大小写

2. **主题样式丢失**:

   - 检查主题 submodule 是否正确初始化
   - 运行 `git submodule update --init --recursive`

3. **404 错误**:

   - 检查 baseurl 配置
   - 确认页面 Front Matter 中没有 `draft: true`

4. **部署失败**:
   - 查看 GitHub Actions 日志
   - 检查 GitHub Secrets 配置
   - 验证服务器 SSH 连接

### 日志位置

- **本地开发**: 终端输出
- **GitHub Actions**: https://github.com/Perlou/go-blog/actions
- **服务器容器**: `docker-compose logs -f`
- **Nginx 日志**: `/var/log/nginx/` (容器内)

## 参考资源

- [Hugo 官方文档](https://gohugo.io/documentation/)
- [Hugo Theme Stack 文档](https://stack.jimmycai.com/)
- [Markdown 语法参考](https://www.markdownguide.org/)
- [TOML 格式参考](https://toml.io/)
- [Giscus 官网](https://giscus.app/)

## 重要提醒

1. **不要修改主题源码**: 所有自定义应通过覆盖的方式进行
2. **使用版本控制**: 所有修改都应提交到 Git
3. **配置 baseURL**: 本地和生产环境使用不同的 baseURL
4. **图片优化**: 上传前压缩图片以提升加载速度
5. **测试后部署**: 本地测试无误后再发布到生产环境
6. **备份重要数据**: 定期备份 `content/` 和配置文件

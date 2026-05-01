# CLAUDE.md

> 这份文件给 Claude（或任何 AI 协作者）看。当你接手 Perlou's Blog 这个
> 项目时，按下面的约束工作，不要绕过。

## 项目是什么

[Perlou's Blog](https://perlou.top) —— 一个**基于 Hugo + Stack 主题**的中文
个人技术博客。主要分享 AI 工程、全栈开发、工具与实践类内容。Docker 容器化
部署，Cloudflare CDN 加速，Giscus 评论。

如果你不熟 Hugo 或 Stack 主题，先扫一眼 [README.md](./README.md) 与
[`.agent/rules.md`](./.agent/rules.md)。再看你接到的具体任务对应的 doc。

---

## 三条不可妥协的边界

### 1. 不要修改 `themes/` 下的主题源码

`themes/hugo-theme-stack/` 是 git submodule。任何样式 / 布局 / 行为定制
**必须通过 `layouts/` override 实现**：

```
layouts/partials/<path>.html
```

会自动覆盖主题中相同路径的文件。覆盖时建议把主题原始文件复制到 `layouts/`
后再改，并保留注释说明改动原因。

判断标准：**如果删掉 `themes/`，重新 `git submodule update --init`，所有
你的定制还在。** 如果不在，说明你改了主题源码。

### 2. 封面图必须放 `assets/images/bg/`，front matter 走 `/images/bg/<file>.webp`

不能用绝对 URL、不能放 `static/`、不能用 jpg/png。

格式：**WebP**，质量 80，尺寸 1024×1024 或 1920×1920（与现有图保持
一致）。Hugo 会通过 `layouts/partials/helper/image.html` 自动生成 800w
和 1600w 两种 srcset。

新图来源（按推荐顺序）：
1. `./scripts/generate-bg-batch.sh -n 1` —— 用 Nano Banana Pro 随机生成
2. 从 `assets/images/bg/` 现有 30+ 张里挑一张复用
3. 自带图片，cwebp 转码后扔进去

### 3. API key 走 `.env`，不进 git

`.env` 已在 `.gitignore` 里。模板看 `.env.example`。脚本会自动 source。

**永远**不要把 `GEMINI_API_KEY` / `OPENAI_API_KEY` / SSH 密钥 / 服务器
密码等写进任何被 git 追踪的文件。

---

## 文件命名 / 路径约定

| 内容 | 位置 | 命名 |
|---|---|---|
| 博客文章 | `content/post/` | `YYYY-MM-DD-article-title.md` |
| 单页 | `content/page/<name>/` | `index.md` |
| 封面图 | `assets/images/bg/` | `<region>-<landmark>.webp` 或 `<topic>.webp` |
| 文章内图 | `static/images/posts/<slug>/` | 任意 |
| 通用图片 | `static/images/` | 任意 |
| Layout override | `layouts/partials/<原路径>` | 与主题文件同名同路径 |
| AI 协作工件 | `.agent/` | rules / templates |
| 可复用 skill | `skills/<skill-name>/SKILL.md` | 标准 Claude Code skill 格式 |

---

## Front Matter 模板

```toml
+++
date = '2026-05-01T10:00:00+08:00'
draft = false
title = '文章标题'
image = '/images/bg/<your-image>.webp'
categories = ['AI', '技术']
tags = ['Tag1', 'Tag2', 'Tag3']
+++
```

- `date`：ISO 8601 + `+08:00`，与内容创建时刻一致
- `draft`：发布前 `false`，写作中 `true`
- `image`：必填，路径以 `/images/bg/` 开头
- `categories`：1-2 个；常用：`AI` / `技术` / `项目实践` / `随笔` / `工具推荐`
- `tags`：3-5 个，精准（参考已有文章选词，避免造重复标签）

---

## 写作风格红线

详见 [`.agent/blog-writing-prompt-template.md`](./.agent/blog-writing-prompt-template.md)。
摘要：

- **中文优先**，专有名词用英文（如 `LLM` / `RAG` / `Hugo`）
- **结构层次**：用 `##` 二级标题分章，`###` 三级展开
- **代码块必带语言标识**：`​`​`​`python` 而不是裸 `​`​`​`
- **多用 emoji 适度点缀**（🚀 ⚡ 💡 ✨）
- **结尾呼应开头**，技术文章给"延伸阅读"，项目分享给"链接 + Star 邀请"
- **避免**：标题过长、大段无分段文字、代码无注释、结构混乱

---

## 常用命令

### 本地开发

```bash
hugo server                          # 启动开发服务器（http://localhost:1313）
hugo server -D                       # 包括 draft
hugo --minify                        # 构建生产版本（输出 public/）
hugo --cleanDestinationDir            # 清缓存重建
```

### 创建 / 发布

```bash
hugo new content/post/YYYY-MM-DD-title.md    # 新文章
./publish.sh                                  # 一键发布（commit + push + 触发部署）
```

### 封面图生成

```bash
cp .env.example .env                              # 一次性配置 GEMINI_API_KEY
./scripts/generate-bg-batch.sh -n 1               # 随机一张（最常用）
./scripts/generate-bg-batch.sh -n 5               # 随机 5 张
./scripts/generate-bg-batch.sh                     # 跑全部还没生成的
./scripts/generate-bg.sh <name> "<prompt>"        # 指定主题单张
```

### 主题更新

```bash
git submodule update --remote --merge themes/hugo-theme-stack
```

### Docker 本地

```bash
docker build -t go-blog:latest .
docker-compose up -d
docker-compose logs -f
```

---

## 提交信息规范

参考 `.agent/rules.md` 的约定：

| 前缀 | 用途 |
|---|---|
| `post:` | 发布 / 修改文章 |
| `draft:` | 草稿状态保存 |
| `feat:` | 新增功能或脚本 |
| `fix:` | 修复 bug |
| `style:` | 样式 / 布局调整 |
| `docs:` | 更新文档 |
| `config:` | 配置文件修改 |
| `chore:` | 杂项 |

例：

```
post: 发布新文章《Harness Engineering》
feat(scripts): random batch mode + 65-entry global manifest
style: 优化 article-list 的图片占位
```

---

## 工作流速查

### 我要发新文章

1. `hugo new content/post/YYYY-MM-DD-title.md`
2. 写正文（参考 `.agent/blog-writing-prompt-template.md` 的结构模板）
3. 选 / 生成封面图：
   - 复用：从 `assets/images/bg/` 选
   - 新生成：`./scripts/generate-bg-batch.sh -n 1`
4. 填好 front matter（date / image / categories / tags）
5. `hugo server` 本地预览
6. `./publish.sh`

### 我要改样式 / 布局

1. 找到主题中要改的文件：`themes/hugo-theme-stack/layouts/<path>.html`
2. **复制**到 `layouts/<path>.html`
3. 在 layouts/ 副本里改，加注释说明原因
4. `hugo server` 验证
5. `git commit -m "style: <动词>"`

### 我要新增主题外的 partial

1. 直接放 `layouts/partials/<name>.html`
2. 在合适的 layout 里 `{{ partial "<name>" . }}`

### 我要更新主题版本

1. `git submodule update --remote --merge themes/hugo-theme-stack`
2. `hugo server` 跑一遍，看自定义还在不在
3. `git add themes/hugo-theme-stack && git commit -m "chore: bump theme"`

---

## 不要做的事

- ❌ 不要 `git push --force` 到 `main`
- ❌ 不要直接编辑 `themes/hugo-theme-stack/` 下的任何文件
- ❌ 不要把 `.env` / 密钥 / token 进 git
- ❌ 不要把 `public/` / `resources/` / `node_modules` 进 git（已 gitignored）
- ❌ 不要在文章 markdown 里用绝对 URL 引用本站资源（用 `/path` 即可）
- ❌ 不要在 front matter `image` 字段里写不存在的文件（构建会失败）
- ❌ 不要为了字数凑内容；技术文章重质不重量

---

## 当你不确定时

| 问题 | 看哪儿 |
|---|---|
| 项目整体怎么跑 | `README.md` |
| 写作风格 / 模板 | `.agent/blog-writing-prompt-template.md` |
| 项目编码规范 | `.agent/rules.md` |
| 怎么部署 | `docs/DEPLOYMENT.md` |
| 怎么发布 | `docs/PUBLISH_WORKFLOW.md` |
| 图片怎么处理 | `docs/IMAGE_WEBP_GUIDE.md` |
| Cloudflare 怎么配 | `docs/CLOUDFLARE_DASHBOARD_GUIDE.md` |
| Giscus 怎么配 | `docs/GISCUS_SETUP_GUIDE.md` |
| 封面图脚本怎么用 | `scripts/README.md` |
| 怎么写一篇符合风格的文章 | `skills/blog-writing/SKILL.md` |
| 怎么生成封面图 | `skills/blog-bg-image/SKILL.md` |
| 怎么改主题不破坏升级 | `skills/hugo-stack-overrides/SKILL.md` |

---

## 推荐阅读顺序（接手时）

1. `README.md` —— 项目入口
2. `.agent/rules.md` —— 项目规则
3. `.agent/blog-writing-prompt-template.md` —— 写作模板
4. `docs/PUBLISH_WORKFLOW.md` —— 发布流程
5. `scripts/README.md` —— 封面图脚本
6. `skills/` —— 可被加载的 skill（按需）

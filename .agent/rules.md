# Perlou's Blog · 项目规则

> 这份文件给 AI 协作者作为"快速对齐"用。完整的边界与命令在
> [`CLAUDE.md`](../CLAUDE.md)；写作风格规范在
> [`blog-writing-prompt-template.md`](./blog-writing-prompt-template.md)；
> 可被加载的 skill 在 [`skills/`](../skills/)。
>
> **本文件只放"项目特定的事实与决策"**，不重复其它文档已有内容。

---

## 项目定位

| 字段 | 值 |
|---|---|
| 名字 | Perlou's Blog |
| 域名 | [perlou.top](https://perlou.top) |
| 仓库 | [Perlou/go-blog](https://github.com/Perlou/go-blog) |
| 作者 | Perlou ([@Perlou](https://github.com/Perlou)) |
| 座右铭 | slow is fast |
| 语言 | 简体中文（zh-cn） |
| 站点类型 | 个人技术博客 + 偶尔随笔 |
| 主题选题 | AI 工程实践 / Python & 数据 / 全栈开发 / 工具体验 |

---

## 真实选题分布（基于 14 篇已发布文章）

```
AI / Agents / LLM / RAG          ~60%
Python / 数据 / 深度学习          ~20%
项目分享 / 工具实践                ~15%
随笔 / 投资观察                   ~5%
```

不太涉及：纯前端 UI 教程、设计、生活类。AI 协作时遵循同分布，避免给出
跑偏的写作建议。

---

## 真实分类与标签池（按实际频率）

### categories（**只用这几种组合**）

| 组合 | 频次 | 适用 |
|---|---|---|
| `['AI', '技术']` | 最常用 | AI / Agents / RAG / LLM / Claude Code / Harness 等 |
| `['Python', '技术']` | 中等 | Python 库 / NumPy / Pandas / PyTorch |
| `['技术']` | 偶尔 | 通用工具 / 项目分享，非 AI 非 Python |
| `['随笔', '投资']` | 罕见 | 行业观察 / 个人思考 |

**禁止**生造 `技术分享` / `AI开发` / `项目实践` / `工具推荐` / `读书笔记`
等没用过的分类。

### tags 池（按真实使用频率）

**AI 高频**：`AI` / `Agents` / `RAG` / `LLM` / `大模型` / `深度学习`

**AI 主题**：`LangChain` / `LlamaIndex` / `RAGAS` / `Hugging Face` /
`Claude Code` / `Harness` / `OpenClaw`

**Python 主题**：`Python` / `NumPy` / `Pandas` / `PyTorch` / `数据分析`

**项目 / 工具**：`开源项目` / `开发工具` / `效率` / `API测试` /
`Cloudflare` / `React` / `TypeScript`

**随笔类**：`XR` / `Vision Pro` / `Meta Ray-Ban` / `空间计算` / `投资`

**约定**：

- AI 类文章必带 `AI`，几乎必带 `Agents`
- 标签 3-5 个，前两个最具体（如 `LangChain`），后续兜底（如 `AI`、`Agents`）
- 大小写：`AI` / `RAG` / `LLM` 全大写；`React` / `TypeScript` /
  `LangChain` 驼峰；`Python` / `Hugging Face` 标准写法
- 不造单字符标签，不造重复（如同时 `LLM` 和 `大模型` 选一即可）

---

## 真实标题模式

观察到 14 篇的标题分四类：

1. **深度解析 / 完整指南**（最常用）
   - "深入解析 Ragas"
   - "LangChain 深入解析：从零开始的完整指南"
   - "LlamaIndex 深入解析：从零到精通"
   - "Harness Engineering: AI Agent 时代的工程范式革命"

2. **手册 / 速查**
   - "NumPy 手册" / "Pandas 手册" / "PyTorch 手册" / "Hugging Face 速查手册"

3. **全面攻略 / 实战**
   - "Claude Code 全面攻略"
   - "OpenClaw 云服务器部署方案"
   - "常用 RAG 方案解析"

4. **项目分享 / 随笔**
   - "基于 Google Antigravity 开发 Httping：一个轻量级 API 测试工具的诞生"
   - "关于 XR 行业和 AI 智能眼镜的一些观察"
   - "大模型入门：原理、架构与实战思考"

写新文章选标题时，**优先按这 4 种模板套**，避免标题党。

---

## 真实写作风格量化观察

- **平均长度**：3000–8000 字（深度解析类）；800–2000 字（项目分享 / 随笔）
- **结构**：几乎所有技术文都有顶部"目录"section（手动写的 anchor 列表）
- **章节编号**：深度解析类用"一/二/三"中文编号；其它用 markdown ##
- **代码块**：100% 带语言标识（` ```python ` 等）
- **emoji**：适度，集中在标题、章节小标题、列表项首
- **表格**：高频出现于"对比 / 速查 / 配置参数"类描述
- **ASCII 图**：偶尔出现（如目录树、流程图、知识图谱）
- **结尾**：技术文必有"总结 / 一段话精要"；项目分享有"链接 + Star 邀请"

---

## 真实工程化基础设施

| 设施 | 现状 |
|---|---|
| 主题 | Hugo Theme Stack（git submodule）+ `layouts/` override，**禁止改主题源码** |
| 性能 | 响应式图片 srcset / DNS prefetch / CSS preload / 评论懒加载（详见 [README.md](../README.md) `⚡ 性能优化` 节） |
| 部署 | Docker → 阿里云 + Cloudflare CDN，GitHub Actions 触发 |
| 评论 | Giscus（GitHub Discussions）+ 懒加载 |
| 分析 | Google Analytics G-CTWSKJMNN4 |
| 搜索 | Hugo 内置 JSON 索引 |
| 封面图 | AI 生成（Nano Banana Pro）→ WebP；脚本在 [`scripts/`](../scripts/) |
| 协作 skill | [`skills/`](../skills/) 三个 SKILL.md（写作 / 配图 / 主题改造） |

---

## AI 协作时的分工

**Claude / 其它 LLM 助手** 接到任务时按以下顺序找答案：

| 任务 | 主参考 | 辅助 |
|---|---|---|
| 写新博客 | [`blog-writing-prompt-template.md`](./blog-writing-prompt-template.md) | [`skills/blog-writing/SKILL.md`](../skills/blog-writing/SKILL.md) |
| 配封面图 | [`scripts/README.md`](../scripts/README.md) | [`skills/blog-bg-image/SKILL.md`](../skills/blog-bg-image/SKILL.md) |
| 改主题样式 | [`skills/hugo-stack-overrides/SKILL.md`](../skills/hugo-stack-overrides/SKILL.md) | [`README.md`](../README.md) `⚡ 性能优化` |
| 部署 / CI | [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) | [`Dockerfile`](../Dockerfile) / [`nginx.conf`](../nginx.conf) |
| 发布文章 | [`docs/PUBLISH_WORKFLOW.md`](../docs/PUBLISH_WORKFLOW.md) | [`publish.sh`](../publish.sh) |
| 项目边界 / 红线 | [`CLAUDE.md`](../CLAUDE.md) | 本文件 |

---

## 三条红线（与 CLAUDE.md 一致，重复确保）

1. **`themes/` 下任何文件不得修改** —— 改样式走 `layouts/` override
2. **封面图必须 WebP，路径 `/images/bg/<name>.webp`** —— 文件不存在 build 会挂
3. **API key 走 `.env`，永不进 git** —— `.gitignore` 已含 `.env`

---

## 提交信息约定

| 前缀 | 用途 | 示例 |
|---|---|---|
| `post:` | 发布 / 修改文章 | `post: 发布新文章《Harness Engineering》` |
| `draft:` | 草稿状态保存 | `draft: WIP RAG 方案对比` |
| `feat:` | 新增功能 / 脚本 | `feat(scripts): random batch mode` |
| `fix:` | 修复 bug | `fix: 修复评论懒加载在 Safari 失效` |
| `style:` | 样式 / 布局调整 | `style: 优化文章列表移动端间距` |
| `docs:` | 更新文档 | `docs: 补充 Cloudflare 配置指南` |
| `config:` | 配置文件修改 | `config: 调整 Nginx gzip 等级` |
| `chore:` | 杂项 / 主题升级 | `chore: bump theme to v3.30` |

---

## 不要做的事（项目级别）

- ❌ 不要用 `hugo new` 之外的方式新建文章（会缺 archetype 默认字段）
- ❌ 不要在文章 markdown 里直接 `<img>` 远端 URL（不可靠且不会进 srcset）
- ❌ 不要在 front matter 加非约定字段（如 `description`、`summary` 等；
   主题不消费这些）
- ❌ 不要在 `content/page/` 下乱建页面（约定只有 about / archives /
   links / search 四种）
- ❌ 不要为了 SEO 堆关键词标签
- ❌ 不要为了"美观"在文章里加大量 emoji（破坏专业感）
- ❌ 不要修改 `archetypes/default.md` 把模板复杂化
- ❌ 不要在新文章里引入主题不支持的 shortcode
- ❌ 不要把 `docker-compose.yml` 里的端口改掉（生产服务器依赖默认 80）

---

## 故障排查（常见的 4 个）

| 症状 | 排查 |
|---|---|
| 本地预览图片 404 | `image` 字段路径必须 `/images/bg/<file>.webp`，文件须真实存在；不要写 `/static/...` |
| 主题样式丢失 | `git submodule update --init --recursive` |
| 部署失败 | GitHub Actions 日志；常见是 `image` 引用不存在 / front matter 语法错 |
| 文章不显示 | `draft: false` 是否设了；日期是否未来时间 |

---

## 维护节奏（项目级目标）

- 写作频率：每月 1-2 篇深度文，间杂随笔 / 手册
- 主题升级：观望 1-2 个 release 再升，避免破窗
- 性能优化：每季度回测一次 Lighthouse；目标 Performance ≥ 95
- AI 工具：紧跟 Claude Code / Cursor / 主流 LLM 的能力边界（与博客主题
  自然契合）

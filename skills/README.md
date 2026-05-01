# skills/

从 [Perlou's Blog](https://github.com/Perlou/go-blog) 项目中提炼出的可复用
[Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) 集合。

> 每份 SKILL.md 都是**项目无关**的，可以单独拷到 `~/.claude/skills/` 全局
> 加载，或拷到任何其它项目的 `.claude/skills/` 里项目级使用。

---

## 三个 skill

| skill | 何时被自动调用 | 来自 |
|---|---|---|
| [`blog-writing`](./blog-writing/SKILL.md) | 用户说"帮我写一篇博客 / 写个 post"，写中文技术博客 | 本仓库 `.agent/blog-writing-prompt-template.md` 与 14 篇已发布文章的结构总结 |
| [`blog-bg-image`](./blog-bg-image/SKILL.md) | 用户说"配张封面图 / 生成 N 张图"，AI 生成博客封面图 | 本仓库 `scripts/generate-bg*.sh` 与 65 条 manifest 的工程经验 |
| [`hugo-stack-overrides`](./hugo-stack-overrides/SKILL.md) | 用户说"改样式 / 调布局 / 主题升级"，安全定制 Hugo Stack 主题 | 本仓库 `layouts/partials/` 下的 override 实践 |

每个 skill 自带：

- 触发条件（何时应当 / 不应当调用）
- 红线与反模式
- 工作流程
- 概念要点 + 代码模板

---

## 快速安装

### 方式 1：全局（所有项目都可见）

```bash
mkdir -p ~/.claude/skills
cp -r skills/blog-writing ~/.claude/skills/
cp -r skills/blog-bg-image ~/.claude/skills/
cp -r skills/hugo-stack-overrides ~/.claude/skills/
```

之后任何 Claude Code 会话都能在合适时机自动调用这些 skill。

### 方式 2：项目级

```bash
cp -r skills/<name> <other-project>/.claude/skills/
```

仅该项目的会话加载。

### 方式 3：直接粘贴

打开 SKILL.md，把整个文件内容（含 frontmatter）粘到 Codex / Cursor /
ChatGPT / 其它 LLM 助手的 system prompt 里。任何能读 Markdown 的 LLM
都生效。

---

## 三份 skill 的关系

```
                 ┌─── blog-writing ────┐
                 │ 写一篇博客文章        │
                 └─────────┬───────────┘
                           │
              ──── 文章需要封面图 ───→
                           │
                           ▼
                 ┌─── blog-bg-image ───┐
                 │ AI 生成 + WebP 转码   │
                 └─────────────────────┘

                 ┌─ hugo-stack-overrides ─┐
                 │ 调主题 / 性能优化 /     │
                 │ 升级安全               │
                 └────────────────────────┘
```

前两个偏内容创作，最后一个偏工程改造。彼此独立可单选加载。

---

## 这些 skill 教 Claude 做什么

### `blog-writing`

- 按"教程 / 项目分享 / 经验总结 / 随笔 / 深度解析" 5 种类型选择结构模板
- 严格走 Front Matter 规范（date / image / categories / tags）
- 中英文之间自动加空格
- 代码块带语言标识 + 必要注释
- 表格 / ASCII 流程图代替"长列表 + 散文"
- 结尾呼应开头 + 给 takeaway

### `blog-bg-image`

- 提示词按"地点 / 场景 / 光线 / 风格"四要素结构化
- 默认走 Nano Banana Pro，按预算 / 可用性回退
- API key 永远走 `.env`，命令行 env 优先
- 维护 manifest（数据驱动）支持随机批量
- WebP + cwebp 转码 + < 300KB 大小约束
- 文件命名 kebab-case 英文，按 `<region>-<landmark>` 组织

### `hugo-stack-overrides`

- 所有定制走 `layouts/` override，**禁止改 `themes/`**
- override 文件顶部加注释说明改动原因
- 主题升级流程：fetch → update → 跑 hugo server 逐页验证
- 5 个高频 override 模板（响应式图、资源预连接、CSS preload、评论懒加载、GA）
- 7 条反模式 + 修正

---

## 起源与可分享性

这三份 skill 由 Perlou 在维护博客 [perlou.top](https://perlou.top) 的过程
中提炼而成，配合 14+ 篇已发布技术博文反复迭代。

**完全 MIT 许可，自由分享**。复制到任何地方、修改、分发都可以。建议保留
SKILL.md 顶部的 frontmatter 让 Claude Code 能正确识别 skill 身份。

如果你扩展或改进了这些 skill，欢迎 fork 维护你自己的版本。

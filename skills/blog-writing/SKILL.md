---
name: blog-writing
description: Use when writing technical blog posts in Chinese, especially for a Hugo-based personal blog with depth-oriented technical content. Covers front matter conventions, structural templates (technical tutorial / project sharing / experience / essay), writing style red lines, and code block / emoji / cover image conventions. Override the default "Markdown 散文" instinct.
version: 1.0.0
license: MIT
---

# Blog Writing · 中文技术博客写作

> 适用于 Hugo + Stack 主题的中文技术博客（如 [perlou.top](https://perlou.top)）。
> 把"写一篇博客"从凭感觉改成结构化产出。

## 何时调用本 skill

**应当**：

- 用户说"帮我写一篇博客 / 帮我写一篇技术文章 / 写个 post"
- 用户提供主题 + 要点要求生成博客内容
- 在 Hugo 项目（含 `content/post/` 目录）里创建新 markdown 文件
- 需要把已有技术内容转写为博客形态

**不应**：

- 写英文技术文章（本 skill 中文优先）
- 写营销文案 / 推广 / 广告
- 写学术论文 / 严肃白皮书
- 用户明确要求"自由风格 / 散文 / 不要结构化"

---

## 文件命名规范

```
content/post/YYYY-MM-DD-url-friendly-title.md
```

- 日期使用文章创建当天
- title 用英文 / 中文拼音的小写 url-safe 形式（不要纯中文文件名）
- 不要省略日期前缀

例：

```
content/post/2026-05-01-harness-engineering.md
content/post/2025-11-21-react-19-features.md
content/post/2026-03-05-ragas.md
```

---

## Front Matter 规范

```toml
+++
date = '<YYYY-MM-DDThh:mm:ss+08:00>'
draft = false
title = '文章标题'
image = '/images/bg/<cover>.webp'
categories = ['<分类1>']
tags = ['<标签1>', '<标签2>', '<标签3>']
+++
```

| 字段 | 规则 |
|---|---|
| `date` | ISO 8601 + `+08:00`（北京时间）；与文件名日期一致 |
| `draft` | 写作时 `true`；发布时 `false` |
| `title` | 短而吸引（< 25 字优先），避免标题党 |
| `image` | 必填；路径 `/images/bg/<file>.webp`；文件须真实存在 |
| `categories` | 1-2 个；从已有列表选，不要造新词 |
| `tags` | 3-5 个，精准；参考已有文章选词，避免单字符 / 全大写 |

### 常用 categories

- `AI` — AI / 机器学习相关
- `技术` — 通用技术分享
- `项目实践` — 实际项目开发
- `随笔` — 思考与观察
- `工具推荐` — 工具使用经验
- `读书笔记` — 书 / 课程总结

### 常用 tags（参考池）

- 技术栈：`React` / `TypeScript` / `Python` / `Go` / `Rust` / `Hugo`
- AI 主题：`LLM` / `Agents` / `RAG` / `Prompt Engineering` / `OpenAI` / `Claude`
- 工程：`性能优化` / `自动化测试` / `API 测试` / `开源项目` / `最佳实践`
- 其它：`生活` / `随想` / `效率工具` / `开发工具`

---

## 结构模板（按文章类型选）

### A. 技术教程类

```markdown
## 背景 / 问题

直接说明这篇文章解决什么问题、为什么重要。

## 解决方案 / 核心概念

讲清楚核心概念。可以加图示 / ASCII 框图。

## 实践步骤

### 步骤 1：xxx

代码 + 必要解释。

### 步骤 2：xxx

继续。

## 最佳实践

3-5 条要点。

## 注意事项 / 常见坑

3-5 条警示。

## 总结

呼应开头，列 3 条 takeaway。
```

### B. 项目分享类

```markdown
## 项目初衷

为什么做这个、解决什么问题。

## 技术选型

技术栈对照表 + 选型理由。

## 开发历程

### 阶段 1：xxx
### 阶段 2：xxx

## 核心功能展示

截图 / GIF / 代码片段。

## 开发感受 / 反思

学到什么、哪里栽过坑。

## 未来规划

下一步打算。

## 总结

总结 + 项目链接 + Star 邀请。
```

### C. 经验总结类

```markdown
## 背景

发生了什么。

## 问题 / 挑战

具体的痛点。

## 解决方案

### 方法 1
### 方法 2

## 效果对比

数据表格 / Before / After。

## 经验教训

3-5 条 takeaway。

## 总结
```

### D. 随笔类

```markdown
## (开篇营造氛围，2-3 段)

## (展开思考，每节一个角度)

## (收束 + 留白)
```

随笔不强求章节标题；可以纯段落连贯。

### E. 深度解析类（推荐用于核心知识介绍）

参考 [github.com/Perlou/go-blog](https://github.com/Perlou/go-blog) 中
`深入解析 X` 类文章（LangChain / Ragas / Harness Engineering）：

```markdown
## 目录（手动列出，每一节都是 anchor link）

---

## 一、X 是什么？
### 1.1 定义
### 1.2 ...

## 二、为什么需要 X？
### 2.1 ...

## 三、核心概念

## 四、...

## ...

## 十一、参考实现 / 配套代码（如有）

## 十二、总结与学习路径
### 12.1 知识图谱（ASCII 图）
### 12.2 推荐学习路径（按 Week 分阶段）
### 12.3 自我检查清单
### 12.4 参考资源（表格）
### 12.5 一段话作为结尾（金句）
```

这种结构特别适合"我要把 X 这个新概念讲清楚"的技术博客文章。

---

## 写作风格（红线）

### ✅ 应该做的

- **中文优先**，专有名词用英文（`LLM` / `RAG` / `Hugo` / `Zod` 等）
- **二级标题分章** (`##`)，三级展开 (`###`)，必要时四级
- **代码块带语言**：` ```python ` 而不是裸 ` ``` `
- **代码块要有注释**，关键变量名要表意清楚
- **emoji 适度点缀**：🚀 ⚡ 🎨 💡 ✨ 📦 🎯 🛡️ 等
- **结尾呼应开头**：技术文给"延伸阅读"，项目分享给"链接 + Star"
- **中英文之间加空格**："使用 React 开发" 而不是"使用React开发"
- **数字与单位之间加空格**："5 个用例" 而不是"5个用例"
- **代码示例完整可运行**（或显式标注"伪代码"）

### ❌ 避免做的

- ❌ 标题过长（> 25 字）或太"标题党"
- ❌ 大段文字没有分段（一段超过 5 行就该断）
- ❌ 代码块没有语言标识
- ❌ 通篇没有代码 / 例子（除随笔外）
- ❌ 章节失衡（有的 1 行，有的 50 行）
- ❌ 结构混乱、跳跃，没有清晰的脉络
- ❌ 滥用 emoji（每段都来一个 = 没意义）
- ❌ 中英混排不空格（很多博客的死结）

### 引用 / 强调

```markdown
> 这是引用：用于关键金句、外部观点、读者要 takeaway 的话。

**这是粗体**：用于第一次出现的术语 / 章节小标题 / 强调。

`code`：变量名 / 文件路径 / 命令片段。
```

### 表格

用表格代替"长列表 + 每条都两三句解释"的场景：

```markdown
| 维度 | A 方案 | B 方案 |
|---|---|---|
| 性能 | 高 | 中 |
| 复杂度 | 中 | 低 |
| 适用 | 高并发 | 简单业务 |
```

### ASCII 图 / 流程图

教学密度高的概念可以用 ASCII：

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Step 1 │───▶│  Step 2 │───▶│  Step 3 │
└─────────┘    └─────────┘    └─────────┘
```

不要用 mermaid 等渲染依赖（Hugo 站点未必装了渲染插件）。

---

## 封面图

每篇文章必须有封面图（front matter `image` 字段）。三个来源：

1. **AI 生成（推荐）**：调本仓库 `scripts/generate-bg-batch.sh -n 1` 随机生成
   一张精选风景图。详见 `skills/blog-bg-image` skill。
2. **复用现有**：从 `assets/images/bg/` 现有图里挑一张主题相符的。
3. **自带图片**：`cwebp -q 80 input.jpg -o assets/images/bg/<name>.webp`。

不要用：
- 远端 URL（不可靠）
- jpg / png（应转 WebP）
- 截图 / 表情包（不够 hero 感）

---

## 长度建议

| 类型 | 字数 |
|---|---|
| 简短随笔 / 工具速记 | 800–1500 字 |
| 标准技术教程 | 1500–3000 字 |
| 深度解析 / 长文 | 3000–8000 字 |
| 课程级合集 / 完整指南 | 8000–15000+ 字 |

不要为字数凑内容。如果一个主题 1500 字能讲透，就不要扩到 3000。

---

## 工作流程（Claude 应用本 skill 时）

当用户说"帮我写一篇博客"时：

1. **确认元信息**（如未给）：主题、类型（A/B/C/D/E）、面向读者、希望长度
2. **选定结构模板**（按上面 A-E 套用）
3. **建议 categories + tags**（从已有池中选，避免造新词）
4. **生成 front matter**（含 `date`、`draft = false`、`image`、`categories`、`tags`）
5. **写正文**：每节都要有内容（不要留 TODO 占位）
6. **代码块带语言标识** + 关键代码加注释
7. **结尾呼应开头** + takeaway
8. **建议封面图**：
   - 推荐 `./scripts/generate-bg-batch.sh -n 1` 随机生成
   - 或从 `assets/images/bg/` 选一张已有的
9. 提醒用户检查 `image` 字段引用的文件是否真实存在（否则构建失败）

---

## 一段话回到起点

技术博客的价值不在"写得多"，而在"让读者真的学到东西"。**结构决定可读性，
代码决定可信度，例子决定可记忆性。** 三者俱全，文章就有生命力。

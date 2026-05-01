---
name: blog-bg-image
description: Use when generating cover images for blog posts via AI image generation, particularly Google Nano Banana Pro / Gemini / Imagen via AI Studio API. Covers prompt structure, model selection, batch generation flow, .env-based key management, WebP conversion, and the manifest pattern for reusable random generation. Use when the user says "生成封面图" / "给文章配张图" / "AI 画一张".
version: 1.0.0
license: MIT
---

# Blog Background Image Generation

> 用 AI 生成博客封面图的完整流水线。Hugo + Stack 主题最适配，但模式
> 通用于任何静态站点。

## 何时调用本 skill

**应当**：

- 用户写完文章问"配张封面图"
- 用户说"生成 N 张风景图 / 技术主题图"
- 用户要补一个新博客主题没有合适已有图
- 维护一个图片资源库（manifest 形式）
- 需要随机产出符合调性的封面

**不应**：

- 用户要求**真实摄影照片**（AI 生成的不是真照片，应去 Unsplash / Pexels）
- 需要**精确人像 / 名人**（AI 生成有合规风险，且效果差）
- 用户要求**带文字的设计图**（Nano Banana Pro 的文字渲染好，但仍不如 Figma）

---

## 三个核心约定

### 1. 文件格式 / 路径

| 项 | 值 |
|---|---|
| 格式 | **WebP**（不要 jpg/png） |
| 路径 | `assets/images/bg/<name>.webp` |
| 尺寸 | 1024×1024（默认）或 1920×1920；需要横图给 1792×1024 |
| 质量 | 80（cwebp `-q 80`） |
| 大小 | < 300KB（超过就 `-q 75` 重压一次） |

### 2. 命名规范

`<region>-<landmark>.webp` 优先：

- `xinjiang-tianchi.webp`
- `iceland-skogafoss.webp`
- `patagonia-torres-del-paine.webp`

技术主题用 `<topic>.webp`：

- `langchain.webp`
- `vector-database.webp`
- `prompt-engineering.webp`

不带空格、不带中文、kebab-case。

### 3. 提示词结构

```
<location/landmark>。<scene description>。
<lighting / time of day>。
photorealistic, no people
```

四要素：地点、场景、光线、风格。**最后两个 phrase 几乎不变**：
"photorealistic, no people"。

例：

```
Tianchi Lake on Bogda Peak in Tianshan Mountains, Xinjiang, China.
Crystal-clear sapphire alpine lake encircled by ancient pine forests
and snow-capped jagged peaks, mirror reflections.
Golden hour soft light, dramatic cumulus clouds.
Photorealistic, no people.
```

---

## 工具与模型

### 工具链

```bash
brew install webp jq    # 一次性
```

- `webp` 提供 `cwebp` —— PNG/JPG → WebP
- `jq` —— manifest JSON 解析与 shuffle

### 模型选择

| 模型 ID | 别名 | 何时选 |
|---|---|---|
| `gemini-3-pro-image-preview` | **Nano Banana Pro** | 默认；4K 分辨率，画面质量最好 |
| `gemini-3.1-flash-image-preview` | Nano Banana 2 | 速度优先，便宜 |
| `gemini-2.5-flash-image` | Nano Banana | 旧版，仍可用 |
| `imagen-4.0-generate-preview-06-06` | Imagen 4 | 专业图像模型 |
| `imagen-3.0-generate-002` | Imagen 3 GA | 稳定回退 |

**默认走 Nano Banana Pro**，除非：
- 预算敏感（→ Nano Banana 2）
- 该 preview 模型暂时不可用（→ Imagen 3 GA）
- 需要严格的 `personGeneration` 控制（→ Imagen 系列才支持）

### API endpoint 自动分流

```
含 imagen 的模型 → :predict      （instances + parameters）
其它 Gemini 模型 → :generateContent  （contents + generationConfig）
```

脚本会按模型名前缀自动选。

---

## API key 管理（红线）

**永远走 `.env` 文件**：

```bash
# .env.example（committable）
GEMINI_API_KEY=AIza-your-key-here
# IMAGE_MODEL=gemini-3-pro-image-preview  (默认就是这个，可不改)
# IMAGE_ASPECT=1:1
```

```bash
# 实际使用
cp .env.example .env
# 编辑 .env 填入真 key
```

**红线**：

- ❌ 不要把 key 写进 commit message
- ❌ 不要把 `.env` 加进 git（`.gitignore` 必须包含 `.env`）
- ❌ 不要把 key 通过命令行 `export GEMINI_API_KEY=AIza...` 后查 history
- ❌ 不要把 key 贴到对话里给 LLM 看（LLM 会记下来）

**.env 加载逻辑**：

脚本里要这样写（**命令行 env 优先**，`.env` 仅作 fallback）：

```bash
if [ -f "$REPO/.env" ]; then
  while IFS='=' read -r _k _v; do
    [[ "$_k" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${_k// }" ]] && continue
    _k="${_k// }"
    _v="${_v%\"}"; _v="${_v#\"}"; _v="${_v%\'}"; _v="${_v#\'}"
    if [ -z "${!_k:-}" ]; then
      export "$_k=$_v"
    fi
  done < "$REPO/.env"
fi
```

**反模式**（这样写命令行变量会被 .env 覆盖）：

```bash
# ❌ 不要这样
set -a; source "$REPO/.env"; set +a
```

---

## Manifest 模式（核心）

把"批量生成"的提示词集中在一份 JSON 里，复用 + 扩展：

```json
[
  {
    "name": "xinjiang-tianchi",
    "prompt": "Aerial view of Tianchi Lake..."
  },
  {
    "name": "iceland-skogafoss",
    "prompt": "Skógafoss waterfall, southern Iceland..."
  }
]
```

**好处**：

- 提示词是工件 → 进 git → 演化可被 review
- 跨项目复用：把 manifest 拷到新博客就能跑
- 支持随机模式：脚本从中随机抽 N 张
- 自动跳过已存在文件 → 节省 API 费用

**Manifest 大小建议**：30-100 条。少了随机重复多，多了维护负担大。

---

## 标准脚本骨架

### 单图生成器（`generate-bg.sh`）

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# 1. 加载 .env（命令行优先）
# ...

NAME="${1:?用法: $0 <name> <prompt>}"
PROMPT="${2:?...}"
MODEL="${IMAGE_MODEL:-gemini-3-pro-image-preview}"
KEY="${GEMINI_API_KEY:-...}"

# 2. 跳过已存在
OUT_WEBP="$REPO/assets/images/bg/$NAME.webp"
[ -f "$OUT_WEBP" ] && [ "${FORCE:-0}" != "1" ] && exit 0

# 3. 选 endpoint + payload（按模型分流）
if [[ "$MODEL" == imagen* ]]; then
  ENDPOINT=".../v1beta/models/$MODEL:predict?key=$KEY"
  PAYLOAD='{"instances":[{"prompt":"..."}], "parameters":{...}}'
else
  ENDPOINT=".../v1beta/models/$MODEL:generateContent?key=$KEY"
  PAYLOAD='{"contents":[{"parts":[{"text":"..."}]}],"generationConfig":{"responseModalities":["IMAGE"]}}'
fi

# 4. 调 API + 解码 + 转 WebP
RESPONSE=$(curl ...)
echo "$RESPONSE" | jq -r '.candidates[0].content.parts[]?.inlineData.data // .predictions[0].bytesBase64Encoded' \
  | base64 -d > "$TMP_PNG"
cwebp -q 80 -mt "$TMP_PNG" -o "$OUT_WEBP"
```

### 批量调度器（`generate-bg-batch.sh`）

```bash
# 关键：可移植 shuffle（macOS 没 shuf）
ENTRIES=$(echo "$POOL" | awk 'BEGIN{srand()} {print rand()"\t"$0}' | sort -k1,1n | head -n "$COUNT" | cut -f2-)

# 关键：跳过已生成的（除非 FORCE=1）
POOL=$(jq -c '.[]' "$MANIFEST" | while IFS= read -r row; do
  name=$(echo "$row" | jq -r '.name')
  [ ! -f "$REPO/assets/images/bg/$name.webp" ] && echo "$row"
done)
```

完整可运行实现见 [github.com/Perlou/go-blog](https://github.com/Perlou/go-blog)
仓库 `scripts/` 目录。

---

## 用户交互的几种典型场景

### 场景 1：用户写完文章问"给我配张封面图"

```bash
./scripts/generate-bg-batch.sh -n 1   # 随机一张
```

让用户从生成结果里挑一张顺眼的。或问用户主题倾向（自然风光 / 科技感 /
特定地区），按倾向调用单图生成器。

### 场景 2：用户说"准备 5 张冰岛图"

不要硬写 5 个 prompt，先**扩 manifest**：

```bash
# 用户视角：编辑 scripts/landscape-prompts.json，加 5 个 iceland-* 条目
# 然后批量跑
./scripts/generate-bg-batch.sh -n 5
# 或只过滤 iceland 条目（jq pre-filter）
jq '[.[] | select(.name | startswith("iceland-"))]' scripts/landscape-prompts.json \
  | ./scripts/generate-bg-batch.sh /dev/stdin
```

### 场景 3：用户要单张特定主题

```bash
./scripts/generate-bg.sh tibet-yamdrok "Yamdrok sacred lake at sunrise..."
```

帮用户写好提示词（按四要素结构），然后调单图生成器。

---

## 费用估算（决策依据）

| 模型 | 单张约价 | 100 张 |
|---|---|---|
| Nano Banana Pro | $0.05–0.15 | $5–15 |
| Nano Banana 2 | $0.02–0.05 | $2–5 |
| Imagen 4 | $0.04 | $4 |
| Imagen 3 | $0.04 | $4 |

跑前提醒用户大概花费，避免误事故大批量烧钱。

---

## 反模式

| 反模式 | 修正 |
|---|---|
| 提示词写一大段散文 | 按"地点 / 场景 / 光线 / 风格"四要素结构化 |
| 每次都现写 prompt | 维护 manifest，复用 + 演化 |
| 直接 export key 跑命令 | 用 `.env`，让脚本自动加载 |
| `source .env` 覆盖命令行 env | 改为"未设置时才 export"的 fallback 模式 |
| 不跳过已存在的图 | 默认跳过，`FORCE=1` 才覆盖 |
| 生成 PNG 直接用 | 一律 cwebp 转 WebP（响应式 + 体积小） |
| 文件名带空格 / 中文 | 一律 kebab-case 英文 |
| Hugo `image` 字段写不存在的文件 | 生成图后立即检查文件是否真实存在 |

---

## 工作流程（Claude 应用本 skill 时）

1. **确认 key 是否就位**：检查 `.env` 是否存在且有 `GEMINI_API_KEY`
2. **确认工具就位**：`which cwebp jq`
3. **确认 manifest 存在**：`scripts/landscape-prompts.json`（或类似路径）
4. **按用户意图决策**：
   - 随机一张 → `-n 1`
   - 多张 → `-n N`
   - 特定主题 → 单图生成 + 显式 prompt
5. **跑完后验证**：`ls -lh assets/images/bg/*.webp` 确认新图出现
6. **提示用户**：把生成的图名写进文章 front matter `image = '/images/bg/...'`

---

## 一句话精要

**封面图不是装饰，是"hero 句子"的视觉版本。** 把生成流程做成数据驱动
（manifest）+ 流水线（脚本）+ 工件化（commit），就从"每次苦想 prompt"
变成"一行命令完成"。

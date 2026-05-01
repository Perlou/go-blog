# scripts/

工程脚本目录。

| 脚本 | 用途 |
|---|---|
| `generate-bg.sh` | 调 Google AI Studio Images API 生成单张博客封面图 → WebP → `assets/images/bg/` |
| `generate-bg-batch.sh` | 批量跑 `landscape-prompts.json` 里的所有图 |
| `landscape-prompts.json` | 12 张精选风景图的提示词 manifest |
| `test-cdn.sh` | 现有的 CDN 测试脚本（独立用途） |

默认模型：**[Nano Banana Pro](https://blog.google/innovation-and-ai/products/nano-banana-pro/)**（`gemini-3-pro-image-preview`）。

---

## 一次性准备

### 1. 装工具

```bash
brew install webp jq
```

### 2. 配置 API key（推荐 .env 形式）

```bash
cp .env.example .env
# 编辑 .env，填入从 https://aistudio.google.com/app/apikey 申请的 key
```

`.env` 已在仓库 `.gitignore` 里，不会被提交。脚本启动时自动 source 它。

> 也可以走纯环境变量：`export GEMINI_API_KEY=AIza...`，跳过 .env。

---

## 批量生成（最常见）

```bash
./scripts/generate-bg-batch.sh
```

行为：
- 读取 `scripts/landscape-prompts.json`，逐张调 Google AI Studio API
- 默认模型 **Nano Banana Pro**（`gemini-3-pro-image-preview`），1:1 比例
- 已存在的 webp **自动跳过**（节省费用）；想强制覆盖加 `FORCE=1`
- 单张失败不打断后续，结尾汇总 OK / 跳过 / 失败计数

---

## 跑单张

```bash
./scripts/generate-bg.sh tibet-yamdrok "Yamdrok sacred lake at sunrise..."
```

---

## 模型 / 长宽比切换

```bash
# 切到 Nano Banana 2（更便宜更快）
IMAGE_MODEL=gemini-3.1-flash-image-preview ./scripts/generate-bg-batch.sh

# 切到 Imagen 4
IMAGE_MODEL=imagen-4.0-generate-preview-06-06 ./scripts/generate-bg-batch.sh

# 16:9 横图（适合大屏 hero）
IMAGE_ASPECT=16:9 ./scripts/generate-bg-batch.sh
```

支持的环境变量（也可写在 `.env`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `GEMINI_API_KEY` | _必填_ | 也接受 `GOOGLE_API_KEY` / `GOOGLE_AI_STUDIO_KEY` |
| `IMAGE_MODEL` | `gemini-3-pro-image-preview`（Nano Banana Pro） | 见下表 |
| `IMAGE_ASPECT` | `1:1` | `1:1` / `16:9` / `9:16` / `4:3` / `3:4` |
| `PERSON_GENERATION` | `DONT_ALLOW` | `DONT_ALLOW` / `ALLOW_ADULT` / `ALLOW_ALL`（仅 Imagen） |
| `FORCE` | `0` | `1` 时覆盖已存在的 webp |

### 可选模型

| 模型 ID | 别名 | 适用 |
|---|---|---|
| `gemini-3-pro-image-preview` | **Nano Banana Pro** | 默认；4K 分辨率，文字渲染最强 |
| `gemini-3.1-flash-image-preview` | Nano Banana 2 | 速度优先，便宜，批量友好 |
| `gemini-2.5-flash-image` | Nano Banana（旧版） | 仍可用 |
| `imagen-4.0-generate-preview-06-06` | Imagen 4 | 专业图像模型 |
| `imagen-3.0-generate-002` | Imagen 3 GA | 稳定，preview 不可用时回退 |

脚本会**根据模型名自动选 endpoint**：含 `imagen` 走 `:predict`，其它走
`:generateContent`。

---

## 编辑 / 扩展 manifest

往 `landscape-prompts.json` 里加一项：

```json
{
  "name": "kamchatka-volcanoes",
  "prompt": "Active volcano landscape on Kamchatka peninsula, Russia. Steaming Klyuchevskaya Sopka volcano with snow on the slopes, vast tundra in the foreground, photorealistic, dramatic dawn light, no people"
}
```

`name` 不带扩展名（脚本自动加 `.webp`）。提示词建议结构：

```
<location/landmark>。<scene description>。<lighting / time of day>。
photorealistic, no people
```

---

## 错误诊断速查

| 报错 | 排查 |
|---|---|
| `API key not valid` | key 错或失效；从 [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) 重新生成 |
| `model not found` | 模型 ID 错；首选 `gemini-3-pro-image-preview`；preview 模型可能轮换 |
| `403 PERMISSION_DENIED` | 当前账号 / 区域不支持该模型；试 `IMAGE_MODEL=imagen-3.0-generate-002` 回退 |
| `400` 含 `safety` | prompt 触发安全过滤；调整描述（避开人物 / 暴力 / 未成年人 / 名人等） |
| `429` | 触发免费 tier 频率限制；等几分钟或升级到付费 tier |
| 响应里没找到图片数据 | 模型可能不支持 `responseModalities: IMAGE`；换成 imagen-* 系列 |

---

## 费用 / 配额参考

按 Google AI Studio 公开定价（2025）：

| model | 计费方式 |
|---|---|
| Nano Banana Pro (`gemini-3-pro-image-preview`) | $2/M input · $12/M output token；单张约 $0.05–0.15 |
| Nano Banana 2 (`gemini-3.1-flash-image-preview`) | 更便宜的 flash 计价 |
| Imagen 4 / 3 | 单张定价 ~$0.04 |

跑完 12 张默认 manifest 大约 **$1 以内**。具体配额以 [Google AI Studio](https://aistudio.google.com) 当前显示为准。

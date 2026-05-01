# scripts/

工程脚本目录。

| 脚本 | 用途 |
|---|---|
| `generate-bg.sh` | 调 OpenAI Images API 生成单张博客封面图 → WebP → `assets/images/bg/` |
| `generate-bg-batch.sh` | 批量跑 `landscape-prompts.json` 里的所有图 |
| `landscape-prompts.json` | 12 张精选风景图的提示词 manifest（新疆 / 挪威 / 巴塔哥尼亚 / 新西兰 / 法罗群岛 / 西藏 / 冰岛 / 加拿大落基山） |
| `test-cdn.sh` | 现有的 CDN 测试脚本（独立用途） |

---

## 批量生成博客封面图

### 一次性准备

```bash
# 工具：webp 编码器 + jq（manifest 解析）
brew install webp jq
```

### 跑批量（最常见）

```bash
export OPENAI_API_KEY=sk-...
./scripts/generate-bg-batch.sh
```

行为：
- 读取 `scripts/landscape-prompts.json`，逐张调 OpenAI Images API
- 默认模型 `gpt-image-1`，1024×1024，quality=high
- 已存在的 webp **自动跳过**（节省费用）；想强制覆盖加 `FORCE=1`
- 单张失败不打断后续，结尾汇总 OK / 跳过 / 失败计数

### 跑单张

```bash
./scripts/generate-bg.sh tibet-yamdrok "Yamdrok sacred lake at sunrise..."
```

### 模型 / 尺寸切换

```bash
IMAGE_MODEL=dall-e-3 IMAGE_SIZE=1792x1024 ./scripts/generate-bg-batch.sh
```

支持的环境变量：

| 变量 | 默认 | 可选值 |
|---|---|---|
| `OPENAI_API_KEY` | _必填_ | sk-... |
| `IMAGE_MODEL` | `gpt-image-1` | `dall-e-3` |
| `IMAGE_SIZE` | `1024x1024` | `1024x1536` / `1536x1024`（gpt-image-1）；`1024x1024` / `1792x1024` / `1024x1792`（dall-e-3） |
| `IMAGE_QUALITY` | `high` | `medium` / `low` / `auto`（仅 gpt-image-1） |
| `FORCE` | `0` | `1` 时覆盖已存在的 webp |

---

## 编辑 / 扩展 manifest

往 `landscape-prompts.json` 里加一项：

```json
{
  "name": "kamchatka-volcanoes",
  "prompt": "Active volcano landscape on Kamchatka peninsula, Russia. Steaming Klyuchevskaya Sopka volcano with snow on the slopes, vast tundra in the foreground, photorealistic, dramatic dawn light, no people"
}
```

`name` 不带扩展名（自动加 `.webp`）。提示词建议结构：

```
<location/landmark>。<scene description>。<lighting / time of day>。photorealistic, no people
```

---

## 费用估算

按 OpenAI 公开定价（2025）：

| model | 1024×1024 standard | 1024×1024 HD |
|---|---|---|
| `gpt-image-1` quality=high | ~$0.07 | — |
| `gpt-image-1` quality=medium | ~$0.04 | — |
| `dall-e-3` standard | $0.040 | — |
| `dall-e-3` hd | — | $0.080 |

跑完 12 张默认 manifest（gpt-image-1 high）≈ $0.84。

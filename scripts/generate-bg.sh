#!/usr/bin/env bash
# 调 Google AI Studio Images API 生成单张博客封面图，转成 WebP 落到
# assets/images/bg/。批量场景见 scripts/generate-bg-batch.sh。
#
# 用法：
#   1. cp .env.example .env
#   2. 编辑 .env，填入 GEMINI_API_KEY
#   3. ./scripts/generate-bg.sh <name> <prompt>
#
#   或直接环境变量：
#      GEMINI_API_KEY=AIza... ./scripts/generate-bg.sh <name> <prompt>
#
# 例：
#   ./scripts/generate-bg.sh tibet-yamdrok "Yamdrok sacred lake at sunrise..."
#
# 环境变量（也可写在 .env 里）：
#   GEMINI_API_KEY        必填（也接受 GOOGLE_API_KEY / GOOGLE_AI_STUDIO_KEY）
#   IMAGE_MODEL           gemini-3-pro-image-preview (默认，即 Nano Banana Pro)
#                         其它选项见 .env.example
#                         脚本根据模型名自动选 :predict（Imagen）或
#                         :generateContent（Gemini / Nano Banana 系列）端点
#   IMAGE_ASPECT          1:1 (默认) | 16:9 | 9:16 | 4:3 | 3:4
#   PERSON_GENERATION     DONT_ALLOW (默认) | ALLOW_ADULT | ALLOW_ALL
#                         （仅 Imagen 系列生效；Gemini 由提示词控制）
#   FORCE                 1 时强制覆盖已有 webp（否则跳过）
set -euo pipefail

# ---------- 自动加载仓库根目录的 .env ----------
# 只把 .env 里的变量当作"未设置时的默认值"——命令行直接传的环境变量
# 优先级更高，避免临时 export 被 .env 覆盖。
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$REPO/.env" ]; then
  while IFS='=' read -r _k _v; do
    [[ "$_k" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${_k// }" ]] && continue
    _k="${_k// }"           # 去 key 两侧空格
    # 去 value 两侧的成对引号
    _v="${_v%\"}"; _v="${_v#\"}"; _v="${_v%\'}"; _v="${_v#\'}"
    if [ -z "${!_k:-}" ]; then
      export "$_k=$_v"
    fi
  done < "$REPO/.env"
  unset _k _v
fi

NAME="${1:?用法: $0 <name-without-extension> <prompt>}"
PROMPT="${2:?用法: $0 <name-without-extension> <prompt>}"
MODEL="${IMAGE_MODEL:-gemini-3-pro-image-preview}"
ASPECT="${IMAGE_ASPECT:-1:1}"
PERSON="${PERSON_GENERATION:-DONT_ALLOW}"

# 三个变量名都接受，按优先级
KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-${GOOGLE_AI_STUDIO_KEY:-}}}"

OUT_DIR="$REPO/assets/images/bg"
OUT_WEBP="$OUT_DIR/$NAME.webp"

# ---------- 前置检查 ----------
if [ -z "$KEY" ]; then
  echo "错误：未找到 GEMINI_API_KEY。" >&2
  echo "提示：cp .env.example .env，然后编辑 .env 填入你的 key。" >&2
  exit 1
fi
command -v cwebp >/dev/null || { echo "错误：未安装 cwebp。macOS: brew install webp" >&2; exit 1; }
command -v jq >/dev/null    || { echo "错误：未安装 jq。macOS: brew install jq" >&2; exit 1; }

if [ -f "$OUT_WEBP" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "↻ 跳过：$OUT_WEBP 已存在（设 FORCE=1 覆盖）"
  exit 0
fi

mkdir -p "$OUT_DIR"

# ---------- 选 endpoint + 构造 payload ----------
BASE="https://generativelanguage.googleapis.com/v1beta/models/$MODEL"

if [[ "$MODEL" == imagen* ]]; then
  ENDPOINT="$BASE:predict?key=$KEY"
  PAYLOAD=$(jq -n \
    --arg prompt "$PROMPT" \
    --arg aspect "$ASPECT" \
    --arg person "$PERSON" \
    '{
      instances: [{prompt: $prompt}],
      parameters: {
        sampleCount: 1,
        aspectRatio: $aspect,
        personGeneration: $person
      }
    }')
  echo "→ $MODEL · aspect=$ASPECT · person=$PERSON"
else
  # Gemini 多模态（Nano Banana / Nano Banana Pro 等）
  ENDPOINT="$BASE:generateContent?key=$KEY"
  PAYLOAD=$(jq -n \
    --arg prompt "$PROMPT" \
    --arg aspect "$ASPECT" \
    '{
      contents: [{
        parts: [{text: $prompt}]
      }],
      generationConfig: {
        responseModalities: ["IMAGE"],
        imageConfig: {aspectRatio: $aspect}
      }
    }')
  echo "→ $MODEL · aspect=$ASPECT"
fi

# ---------- 调 API ----------
RESPONSE=$(curl -sS "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

if echo "$RESPONSE" | jq -e '.error' >/dev/null 2>&1; then
  echo "× API 错误：" >&2
  echo "$RESPONSE" | jq -r '.error.message // .error' >&2
  exit 2
fi

# ---------- 抽取图片 b64（兼容两种响应形态） ----------
TMP_PNG="$(mktemp -t blog-bg.XXXXXX).png"
trap 'rm -f "$TMP_PNG"' EXIT

if echo "$RESPONSE" | jq -e '.predictions[0].bytesBase64Encoded' >/dev/null 2>&1; then
  # Imagen 路径
  echo "$RESPONSE" | jq -r '.predictions[0].bytesBase64Encoded' | base64 -d > "$TMP_PNG"
elif echo "$RESPONSE" | jq -e '.candidates[0].content.parts[]?.inlineData.data' >/dev/null 2>&1; then
  # Gemini 路径（找到第一个 inlineData）
  echo "$RESPONSE" | jq -r 'first(.candidates[0].content.parts[] | select(.inlineData) | .inlineData.data)' | base64 -d > "$TMP_PNG"
else
  echo "× 响应里没找到图片数据：" >&2
  echo "$RESPONSE" | head -c 800 >&2
  echo >&2
  exit 3
fi

# ---------- 转 WebP ----------
cwebp -q 80 -mt "$TMP_PNG" -o "$OUT_WEBP" 2>&1 | tail -1

SIZE_BYTES=$(stat -f%z "$OUT_WEBP" 2>/dev/null || stat -c%s "$OUT_WEBP")
echo "✓ $OUT_WEBP ($((SIZE_BYTES / 1024)) KB)"

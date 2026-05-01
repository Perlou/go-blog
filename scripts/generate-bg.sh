#!/usr/bin/env bash
# 调用 OpenAI Images API 生成单张博客封面图，转成 WebP 落到
# assets/images/bg/。批量场景见 scripts/generate-bg-batch.sh。
#
# 用法：
#   OPENAI_API_KEY=sk-... ./scripts/generate-bg.sh <name> <prompt>
#
# 例：
#   ./scripts/generate-bg.sh tibet-yamdrok "Yamdrok sacred lake at sunrise..."
#
# 环境变量：
#   OPENAI_API_KEY        必填
#   IMAGE_MODEL           gpt-image-1 (默认) | dall-e-3
#   IMAGE_SIZE            1024x1024 (默认) | 1024x1536 | 1536x1024
#   IMAGE_QUALITY         high (默认) | medium | low | auto
#   FORCE                 1 时强制覆盖已有 webp（否则跳过）
set -euo pipefail

NAME="${1:?用法: $0 <name-without-extension> <prompt>}"
PROMPT="${2:?用法: $0 <name-without-extension> <prompt>}"
MODEL="${IMAGE_MODEL:-gpt-image-1}"
SIZE="${IMAGE_SIZE:-1024x1024}"
QUALITY="${IMAGE_QUALITY:-high}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO/assets/images/bg"
OUT_WEBP="$OUT_DIR/$NAME.webp"

# ---------- 前置检查 ----------
[ -z "${OPENAI_API_KEY:-}" ] && { echo "错误：未设置 OPENAI_API_KEY 环境变量。" >&2; exit 1; }
command -v cwebp >/dev/null || { echo "错误：未安装 cwebp。macOS: brew install webp" >&2; exit 1; }
command -v jq >/dev/null    || { echo "错误：未安装 jq。macOS: brew install jq" >&2; exit 1; }

if [ -f "$OUT_WEBP" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "↻ 跳过：$OUT_WEBP 已存在（设 FORCE=1 覆盖）"
  exit 0
fi

mkdir -p "$OUT_DIR"

# ---------- 调 OpenAI Images API ----------
echo "→ 请求 $MODEL · $SIZE · quality=$QUALITY"

PAYLOAD=$(jq -n \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  --arg size "$SIZE" \
  --arg quality "$QUALITY" \
  '{model:$model, prompt:$prompt, size:$size, quality:$quality, n:1}')

RESPONSE=$(curl -sS https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "$PAYLOAD")

if echo "$RESPONSE" | jq -e '.error' >/dev/null 2>&1; then
  echo "× API 错误：" >&2
  echo "$RESPONSE" | jq -r '.error.message' >&2
  exit 2
fi

# ---------- 解码 + 转 WebP ----------
TMP_PNG="$(mktemp -t blog-bg.XXXXXX).png"
trap 'rm -f "$TMP_PNG"' EXIT

# gpt-image-1 默认 b64_json；DALL-E 3 默认 url。两种都处理。
if echo "$RESPONSE" | jq -e '.data[0].b64_json' >/dev/null 2>&1; then
  echo "$RESPONSE" | jq -r '.data[0].b64_json' | base64 -d > "$TMP_PNG"
elif echo "$RESPONSE" | jq -e '.data[0].url' >/dev/null 2>&1; then
  IMG_URL=$(echo "$RESPONSE" | jq -r '.data[0].url')
  curl -sS "$IMG_URL" -o "$TMP_PNG"
else
  echo "× 响应里既没 b64_json 也没 url：" >&2
  echo "$RESPONSE" | head -c 500 >&2
  exit 3
fi

cwebp -q 80 -mt "$TMP_PNG" -o "$OUT_WEBP" 2>&1 | tail -1

SIZE_BYTES=$(stat -f%z "$OUT_WEBP" 2>/dev/null || stat -c%s "$OUT_WEBP")
echo "✓ $OUT_WEBP ($((SIZE_BYTES / 1024)) KB)"

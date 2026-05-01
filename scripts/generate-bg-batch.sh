#!/usr/bin/env bash
# 批量生成 scripts/landscape-prompts.json 列出的所有封面图。
# 每条调用一次 generate-bg.sh，单张失败不影响后续。
#
# 用法：
#   OPENAI_API_KEY=sk-... ./scripts/generate-bg-batch.sh
#   OPENAI_API_KEY=sk-... ./scripts/generate-bg-batch.sh path/to/other.json
#   FORCE=1 ./scripts/generate-bg-batch.sh   # 强制覆盖已有图
#
# 默认会自动跳过已经存在的 webp（节省 API 费用）。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:-$REPO/scripts/landscape-prompts.json}"
GEN="$REPO/scripts/generate-bg.sh"

[ -f "$MANIFEST" ] || { echo "manifest 不存在: $MANIFEST" >&2; exit 1; }
[ -x "$GEN" ]      || { echo "未找到或不可执行: $GEN" >&2; exit 1; }

COUNT=$(jq 'length' "$MANIFEST")
echo "→ 共 $COUNT 张待生成（manifest: $MANIFEST）"
echo "→ FORCE=${FORCE:-0}（=1 强制覆盖）"
echo

OK=0
FAIL=0
SKIP=0
INDEX=0

# 用 process substitution 避免 while 在 subshell 里丢失计数
while IFS=$'\t' read -r NAME PROMPT; do
  INDEX=$((INDEX + 1))
  printf '──── [%d/%d] %s ────\n' "$INDEX" "$COUNT" "$NAME"
  if "$GEN" "$NAME" "$PROMPT"; then
    if [ -f "$REPO/assets/images/bg/$NAME.webp" ]; then
      OK=$((OK + 1))
    else
      SKIP=$((SKIP + 1))
    fi
  else
    FAIL=$((FAIL + 1))
  fi
  echo
done < <(jq -r '.[] | "\(.name)\t\(.prompt)"' "$MANIFEST")

echo "════════════════════════════════════"
echo "  完成 OK=$OK  跳过=$SKIP  失败=$FAIL  总=$COUNT"
echo "════════════════════════════════════"
echo "  ls -lh assets/images/bg/*.webp"

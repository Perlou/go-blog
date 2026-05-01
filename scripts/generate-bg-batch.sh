#!/usr/bin/env bash
# 批量生成 manifest 里的封面图，支持随机模式。
#
# 用法：
#   ./scripts/generate-bg-batch.sh                    跑全部还没生成的图
#   ./scripts/generate-bg-batch.sh -n 5               随机抽 5 张
#   ./scripts/generate-bg-batch.sh -n 1               随机一张（最常用）
#   ./scripts/generate-bg-batch.sh -n 3 path/to.json  用其它 manifest
#   ./scripts/generate-bg-batch.sh path/to.json       全部跑（指定 manifest）
#
# 默认会自动跳过 assets/images/bg/ 里已存在的同名 webp（节省 API 费用）。
# FORCE=1 强制覆盖；同时随机模式会把已存在的图重新放回抽取池。
#
# 模型 / 长宽比 / 是否允许出现人 见 generate-bg.sh 顶部注释或 .env.example。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# 加载 .env，但命令行 env 优先（与 generate-bg.sh 保持一致）
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
  unset _k _v
fi

# ---------- 参数解析 ----------
COUNT=""
MANIFEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--count)
      COUNT="${2:?-n 后必须跟数字}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)
      MANIFEST="$1"
      shift
      ;;
  esac
done

MANIFEST="${MANIFEST:-$REPO/scripts/landscape-prompts.json}"
GEN="$REPO/scripts/generate-bg.sh"

[ -f "$MANIFEST" ] || { echo "manifest 不存在: $MANIFEST" >&2; exit 1; }
[ -x "$GEN" ]      || { echo "未找到或不可执行: $GEN" >&2; exit 1; }
command -v jq >/dev/null || { echo "错误：未安装 jq。macOS: brew install jq" >&2; exit 1; }

if [ -n "$COUNT" ] && ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：-n 必须是正整数（收到: $COUNT）" >&2
  exit 1
fi

TOTAL=$(jq 'length' "$MANIFEST")

# ---------- 构建 pool（FORCE=1 时含全部，否则过滤掉已存在的） ----------
if [ "${FORCE:-0}" = "1" ]; then
  POOL=$(jq -c '.[]' "$MANIFEST")
else
  POOL=$(jq -c '.[]' "$MANIFEST" | while IFS= read -r row; do
    name=$(echo "$row" | jq -r '.name')
    [ ! -f "$REPO/assets/images/bg/$name.webp" ] && echo "$row"
  done)
fi

POOL_SIZE=$(if [ -z "$POOL" ]; then echo 0; else echo "$POOL" | grep -c '^'; fi)

# ---------- 决定要跑哪些条目 ----------
if [ -n "$COUNT" ]; then
  if [ "$POOL_SIZE" -eq 0 ]; then
    echo "✓ manifest 共 $TOTAL 条，全部已生成。设 FORCE=1 重新生成。"
    exit 0
  fi
  if [ "$COUNT" -gt "$POOL_SIZE" ]; then
    echo "→ 请求 $COUNT 张，但池子里只有 $POOL_SIZE 张未生成；改为生成 $POOL_SIZE 张。"
    COUNT=$POOL_SIZE
  fi
  # awk 实现的可移植 shuffle（macOS 没有 shuf）
  ENTRIES=$(echo "$POOL" | awk 'BEGIN{srand()} {print rand()"\t"$0}' | sort -k1,1n | head -n "$COUNT" | cut -f2-)
  TO_RUN=$COUNT
  echo "🎲 随机模式: 从 manifest $TOTAL 条中抽 $COUNT 张（已生成 $((TOTAL - POOL_SIZE)) 张被排除在抽取池外）"
else
  if [ "$POOL_SIZE" -eq 0 ]; then
    echo "✓ manifest 共 $TOTAL 条，全部已生成。设 FORCE=1 重新生成。"
    exit 0
  fi
  ENTRIES=$POOL
  TO_RUN=$POOL_SIZE
  echo "→ 全部模式: 跑 $POOL_SIZE 张未生成的（manifest 共 $TOTAL）"
fi

echo "→ 模型: ${IMAGE_MODEL:-gemini-3-pro-image-preview}  长宽比: ${IMAGE_ASPECT:-1:1}"
echo "→ FORCE=${FORCE:-0}"
echo

# ---------- 执行 ----------
OK=0
FAIL=0
INDEX=0

while IFS= read -r row; do
  [ -z "$row" ] && continue
  INDEX=$((INDEX + 1))
  NAME=$(echo "$row" | jq -r '.name')
  PROMPT=$(echo "$row" | jq -r '.prompt')
  printf '──── [%d/%d] %s ────\n' "$INDEX" "$TO_RUN" "$NAME"
  if "$GEN" "$NAME" "$PROMPT"; then
    OK=$((OK + 1))
  else
    FAIL=$((FAIL + 1))
    echo "× $NAME 失败，继续下一张"
  fi
  echo
done <<< "$ENTRIES"

echo "════════════════════════════════════"
echo "  完成 OK=$OK  失败=$FAIL  共=$TO_RUN"
echo "════════════════════════════════════"
echo "  ls -lh assets/images/bg/*.webp"

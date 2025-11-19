#!/bin/bash

# 博客发布脚本
# 使用方法: ./publish.sh

set -e  # 遇到错误立即退出

echo "📝 检查是否有改动..."
if [[ -z $(git status -s) ]]; then
    echo "✅ 没有需要提交的改动"
    exit 0
fi

echo ""
echo "📋 当前改动："
git status -s

echo ""
echo "💬 请输入提交信息 (例如: post: 发布新文章):"
read -r commit_msg

if [[ -z "$commit_msg" ]]; then
    echo "❌ 提交信息不能为空"
    exit 1
fi

echo ""
echo "📦 添加所有改动..."
git add .

echo "💾 提交改动..."
git commit -m "$commit_msg"

echo "🚀 推送到 GitHub..."
git push

echo ""
echo "🔄 触发部署..."

# 检查是否安装了 gh CLI
if command -v gh &> /dev/null; then
    echo "   使用 GitHub CLI 触发部署..."
    if gh workflow run deploy.yml 2>/dev/null; then
        echo "   ✅ 部署已触发"
    else
        echo "   ⚠️  自动触发失败，请手动触发："
        echo "   https://github.com/Perlou/go-blog/actions/workflows/deploy.yml"
    fi
else
    echo "   ⚠️  未安装 GitHub CLI (gh)"
    echo "   请访问以下链接手动触发部署："
    echo "   https://github.com/Perlou/go-blog/actions/workflows/deploy.yml"
    echo ""
    echo "   💡 安装 GitHub CLI 以实现自动触发："
    echo "   brew install gh"
fi

echo ""
echo "✅ 发布完成！"
echo ""
echo "📊 查看部署进度："
echo "   https://github.com/Perlou/go-blog/actions"
echo ""
echo "🌐 博客地址："
echo "   https://perlou.top"
echo ""
echo "⏰ 预计 2-3 分钟后部署完成"

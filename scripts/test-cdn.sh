#!/bin/bash

echo "🧪 Cloudflare CDN 验证测试"
echo "================================"
echo ""

DOMAIN="perlou.top"
TEST_URLS=(
    "https://${DOMAIN}/images/covers/langchain.jpg"
    "https://${DOMAIN}/index.html"
)

for url in "${TEST_URLS[@]}"; do
    echo "📍 测试: $url"
    
    response=$(curl -sI "$url")
    
    # 检查 Cloudflare 头
    if echo "$response" | grep -q "cloudflare"; then
        echo "  ✅ Cloudflare: 已启用"
    else
        echo "  ❌ Cloudflare: 未检测到"
    fi
    
    # 检查缓存状态
    cache_status=$(echo "$response" | grep -i "cf-cache-status" | cut -d' ' -f2 | tr -d '\r')
    if [ -n "$cache_status" ]; then
        echo "  ✅ Cache Status: $cache_status"
    fi
    
    # 检查 Ray ID（Cloudflare 请求 ID）
    ray_id=$(echo "$response" | grep -i "cf-ray" | cut -d' ' -f2 | tr -d '\r')
    if [ -n "$ray_id" ]; then
        echo "  ✅ CF-Ray: $ray_id"
    fi
    
    # 检查 Cache-Control
    cache_control=$(echo "$response" | grep -i "cache-control" | cut -d' ' -f2- | tr -d '\r')
    if [ -n "$cache_control" ]; then
        echo "  ℹ️  Cache-Control: $cache_control"
    fi
    
    echo ""
done

echo "================================"
echo "✅ 测试完成"
echo ""
echo "💡 提示："
echo "  - 如果看到 'cf-cache-status: HIT'，说明 CDN 缓存已生效"
echo "  - 第一次访问可能是 'MISS'，这是正常的"
echo "  - 多次运行此脚本，应该看到 'HIT' 状态"

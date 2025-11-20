# Cloudflare 配置指南

本指南帮助你配置 Cloudflare CDN 加速。

## 添加网站

### 1. 注册并添加站点

1. 访问：https://dash.cloudflare.com/sign-up（注册账号）
2. 登录后点击 `Add a Site`
3. 输入域名：`perlou.top`
4. 选择 **Free 免费计划**

### 2. 检查 DNS 记录

确保有以下记录，且 Proxy Status 为 **Proxied**（橙色云朵）：

| Type | Name | Content   | Proxy      |
| ---- | ---- | --------- | ---------- |
| A    | @    | 服务器 IP | ☁️ Proxied |
| A    | www  | 服务器 IP | ☁️ Proxied |

## 修改 DNS 服务器

### 1. 获取 Cloudflare DNS

记下 Cloudflare 提供的两个 DNS 服务器：

```
nameserver 1: xxx.ns.cloudflare.com
nameserver 2: xxx.ns.cloudflare.com
```

### 2. 修改域名 DNS

**阿里云**：

1. 登录阿里云控制台
2. `域名` → `域名列表` → `管理`
3. `DNS 修改` → 输入 Cloudflare DNS
4. 确定

**等待生效**：5-30 分钟（会收到激活邮件）

## SSL/TLS 配置

1. `SSL/TLS` → `Overview`
2. 加密模式：**Full (strict)**

3. `SSL/TLS` → `Edge Certificates`
4. 开启以下选项：
   - ✅ Always Use HTTPS
   - ✅ Automatic HTTPS Rewrites
   - ✅ TLS 1.3

## 性能优化

### 1. 基础优化

`Speed` → `Optimization`：

- ✅ Auto Minify：勾选 JavaScript、CSS、HTML
- ✅ Brotli
- ✅ Early Hints

### 2. 缓存配置

`Caching` → `Configuration`：

| 选项              | 设置     |
| ----------------- | -------- |
| Caching Level     | Standard |
| Browser Cache TTL | 4 hours  |

### 3. Page Rules（重要）

`Rules` → `Page Rules` → `Create Page Rule`

**规则 1：图片缓存**

URL Pattern:

```
perlou.top/images/*
```

Settings:

- Cache Level: Cache Everything
- Edge Cache TTL: 1 month
- Browser Cache TTL: 1 year

**规则 2：静态资源**（可选）

URL Pattern:

```
perlou.top/css/*
```

Settings:

- Cache Level: Cache Everything
- Edge Cache TTL: 1 month
- Browser Cache TTL: 1 year

## 网络配置

`Network`：

- ✅ HTTP/2
- ✅ HTTP/3 (QUIC)
- ✅ 0-RTT Connection Resumption

## 安全配置

`Security` → `Settings`：

- Security Level: **Medium**
- ✅ Bot Fight Mode

## 验证配置

### 测试 CDN

打开浏览器开发者工具（F12）→ Network

访问：`https://perlou.top/images/covers/xxx.jpg`

查看响应头：

```
server: cloudflare
cf-cache-status: HIT
```

看到 `HIT` 表示缓存生效。

## 维护操作

### 清除缓存

`Caching` → `Configuration`：

- Purge Everything：清除所有缓存
- Custom Purge：清除特定 URL

### 开发模式

需要频繁测试时开启（暂停缓存 3 小时）：
`Caching` → `Configuration` → Development Mode

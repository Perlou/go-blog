# Cloudflare Dashboard 配置步骤指南

本指南将帮助你在 Cloudflare Dashboard 中完成所有必要的 CDN 配置。

---

## 📋 准备工作

开始之前，请确保：

- ✅ 已优化 Nginx 配置（已完成）
- ⏳ Cloudflare 账号（如果没有，请先注册：https://dash.cloudflare.com/sign-up）
- ⏳ 可以访问域名注册商管理后台

---

## 🚀 步骤 1：注册并添加网站

### 1.1 注册 Cloudflare 账号

如果还没有账号：

1. 访问：https://dash.cloudflare.com/sign-up
2. 输入邮箱和密码
3. 验证邮箱

### 1.2 添加网站

1. 登录 Cloudflare Dashboard
2. 点击 **「添加站点」** 或 **「Add a Site」**
3. 输入域名：`perlou.top`
4. 点击 **「添加站点」**

### 1.3 选择计划

- 选择 **Free 免费计划**（$0/月）
- 点击 **「继续」**

### 1.4 扫描 DNS 记录

Cloudflare 会自动扫描你的现有 DNS 记录。

**检查以下记录是否存在**：

| Type | Name | Content         | Proxy Status           |
| ---- | ---- | --------------- | ---------------------- |
| A    | @    | [你的服务器 IP] | ☁️ Proxied（橙色云朵） |
| A    | www  | [你的服务器 IP] | ☁️ Proxied（橙色云朵） |

⚠️ **重要**：确保 Proxy Status 为 **Proxied**（显示为橙色云朵），这样流量才会经过 CDN。

如果没有这些记录，手动添加：

1. 点击 **「添加记录」**
2. 填写上述信息
3. 确保开启 Proxy（橙色云朵）

完成后点击 **「继续」**。

---

## 🔧 步骤 2：修改 DNS 服务器

### 2.1 获取 Cloudflare DNS 服务器

Cloudflare 会显示两个 DNS 服务器地址，类似：

```
nameserver 1: blake.ns.cloudflare.com
nameserver 2: sara.ns.cloudflare.com
```

📝 **记下这两个地址！**

### 2.2 前往域名注册商修改 DNS

#### 如果你的域名在阿里云：

1. 登录 **阿里云控制台**：https://dc.console.aliyun.com/
2. 进入 **域名** → **域名列表**
3. 找到 `perlou.top`，点击 **「管理」**
4. 点击左侧 **「DNS 修改」**
5. 选择 **「修改 DNS 服务器」**
6. 输入 Cloudflare 提供的两个 DNS：
   - DNS 服务器 1：`blake.ns.cloudflare.com`
   - DNS 服务器 2：`sara.ns.cloudflare.com`
7. 点击 **「确定」**

#### 如果你的域名在腾讯云：

1. 登录腾讯云控制台
2. 进入 **域名注册** → **我的域名**
3. 点击 `perlou.top` 后的 **「管理」**
4. 找到 **「DNS 服务器」**，点击 **「修改」**
5. 输入 Cloudflare 的两个 DNS
6. **保存**

#### 如果你的域名在其他注册商：

通常在域名管理页面找到 **「DNS 服务器」** 或 **「Name Servers」** 设置，替换为 Cloudflare 提供的地址。

### 2.3 等待 DNS 生效

- ⏱️ 通常需要 **5-30 分钟**
- 有时可能需要 **24-48 小时**（取决于注册商）
- Cloudflare 会发邮件通知你激活成功
- 在此期间网站访问正常，无需担心

💡 **提示**：可以在 Cloudflare Dashboard 查看状态，激活成功后会显示 **「Active」**。

---

## 🔒 步骤 3：配置 SSL/TLS

### 3.1 设置加密模式

1. 在 Cloudflare Dashboard 左侧菜单，点击 **SSL/TLS**
2. 在 **Overview** 标签中
3. 选择加密模式：**Full (strict)** ✅

**模式说明**：

- ❌ Off：不加密（不推荐）
- ❌ Flexible：仅 Cloudflare 到访客加密，源站不加密（不推荐）
- ⚠️ Full：Cloudflare 到源站加密，但不验证证书
- ✅ **Full (strict)**：端到端加密 + 验证源站证书（推荐）

### 3.2 强制 HTTPS

1. 点击 **SSL/TLS** → **Edge Certificates**
2. 开启以下选项：

| 选项                         | 状态        | 说明                       |
| ---------------------------- | ----------- | -------------------------- |
| **Always Use HTTPS**         | ✅ ON       | 自动将 HTTP 重定向到 HTTPS |
| **Automatic HTTPS Rewrites** | ✅ ON       | 自动重写混合内容           |
| **Minimum TLS Version**      | **TLS 1.2** | 最低 TLS 版本              |
| **TLS 1.3**                  | ✅ ON       | 启用更快更安全的 TLS 1.3   |

---

## ⚡ 步骤 4：速度优化配置

### 4.1 Auto Minify（自动压缩）

1. 点击 **Speed** → **Optimization**
2. 找到 **Auto Minify** 部分
3. 勾选以下选项：

- ☑️ **JavaScript**
- ☑️ **CSS**
- ☑️ **HTML**

### 4.2 Brotli 压缩

在同一页面，找到 **Brotli** 选项：

- ✅ **ON** - 启用（比 Gzip 压缩效果更好）

### 4.3 Early Hints

- ✅ **ON** - 启用（提前加载资源，加快首屏渲染）

### 4.4 Rocket Loader

⚠️ **暂时不要启用** - 可能会影响某些 JavaScript 功能，建议先测试后再开启。

---

## 💾 步骤 5：缓存配置（核心！）

### 5.1 基础缓存设置

1. 点击 **Caching** → **Configuration**
2. 配置以下选项：

| 选项                  | 设置值       | 说明                     |
| --------------------- | ------------ | ------------------------ |
| **Caching Level**     | **Standard** | 标准缓存级别             |
| **Browser Cache TTL** | **4 hours**  | 浏览器缓存时间           |
| **Crawlers Hint**     | **Off**      | 关闭爬虫提示（节省流量） |

### 5.2 创建 Page Rules（重要！）

Page Rules 是 Cloudflare 最强大的功能之一，可以为不同类型的资源设置精细化缓存策略。

**免费计划限制：最多 3 条 Page Rules**

#### 创建规则步骤：

1. 点击 **Rules** → **Page Rules**
2. 点击 **「Create Page Rule」**

---

#### 🔥 规则 1：图片资源缓存

**URL Pattern**（URL 模式）：

```
perlou.top/images/*
```

**Settings**（设置）：

1. 点击 **「+ Add a Setting」**，添加以下 3 个设置：

| Setting               | Value            |
| --------------------- | ---------------- |
| **Cache Level**       | Cache Everything |
| **Edge Cache TTL**    | 1 month          |
| **Browser Cache TTL** | 1 year           |

2. 点击 **「Save and Deploy」**

**说明**：

- 所有 `/images/` 下的资源都会被 CDN 缓存 1 个月
- 用户浏览器会缓存 1 年
- 极大减少源站流量

---

#### 🔥 规则 2：静态资源缓存

**URL Pattern**：

```
perlou.top/*.css
```

或使用通配符（推荐）：

```
perlou.top/*
```

然后使用 **Cache Key** 配置（Pro 功能，免费版可用简化版）

**更简单的方式（免费版）**：

为每种静态资源类型创建单独规则，或使用一条规则覆盖主要静态资源：

**URL Pattern**：

```
perlou.top/css/*
```

**Settings**：

| Setting               | Value            |
| --------------------- | ---------------- |
| **Cache Level**       | Cache Everything |
| **Edge Cache TTL**    | 1 month          |
| **Browser Cache TTL** | 1 year           |

重复此步骤为 `/js/*` 创建类似规则（如果有 3 条规则限制，可以只保留图片规则）。

---

#### ⚠️ 规则 3：HTML 缓存（可选）

**建议**：如果你经常更新文章内容，**不要**设置 HTML 缓存规则，保留一个规则槽位给未来使用。

如果想缓存 HTML（不常更新内容）：

**URL Pattern**：

```
perlou.top/*.html
```

**Settings**：

| Setting               | Value            |
| --------------------- | ---------------- |
| **Cache Level**       | Cache Everything |
| **Edge Cache TTL**    | 1 hour           |
| **Browser Cache TTL** | 10 minutes       |

---

### 5.3 验证 Page Rules 顺序

确保规则顺序正确（从上到下匹配）：

1. 图片资源规则
2. 静态资源规则
3. HTML 规则（如果有）

可以拖动规则调整顺序。

---

## 🌐 步骤 6：网络配置

1. 点击 **Network**
2. 确认以下选项已启用：

| 选项                            | 状态  | 说明                 |
| ------------------------------- | ----- | -------------------- |
| **HTTP/2**                      | ✅ ON | 默认启用，更快的连接 |
| **HTTP/3 (QUIC)**               | ✅ ON | 最新协议，速度更快   |
| **0-RTT Connection Resumption** | ✅ ON | 加快重复访问速度     |
| **WebSockets**                  | ✅ ON | 支持 WebSocket 连接  |

---

## 🛡️ 步骤 7：安全配置

### 7.1 Security Level

1. 点击 **Security** → **Settings**
2. **Security Level**：选择 **Medium**
   - Low：最宽松
   - **Medium**：平衡（推荐）
   - High：严格
   - Under Attack：遭受攻击时启用

### 7.2 Bot Fight Mode

在同一页面：

- ✅ **Bot Fight Mode**：ON - 免费的 bot 防护

---

## ✅ 步骤 8：验证配置

### 8.1 检查 DNS 激活状态

1. 返回 Cloudflare Dashboard 首页
2. 查看 `perlou.top` 状态
3. 应该显示 **「Active」**（绿色）

### 8.2 测试 CDN 是否生效

打开终端，运行测试脚本：

```bash
cd /Users/perlou/Desktop/personal/go-blog
./scripts/test-cdn.sh
```

**期望输出**：

```
🧪 Cloudflare CDN 验证测试
================================

📍 测试: https://perlou.top/images/covers/langchain.jpg
  ✅ Cloudflare: 已启用
  ✅ Cache Status: HIT
  ✅ CF-Ray: 8xxxxxxxxxxxxx-XXX

📍 测试: https://perlou.top/index.html
  ✅ Cloudflare: 已启用
  ✅ Cache Status: HIT
  ✅ CF-Ray: 8xxxxxxxxxxxxx-XXX

================================
✅ 测试完成
```

如果第一次运行看到 `MISS`，这是正常的（缓存未命中），多运行几次应该看到 `HIT`。

### 8.3 手动浏览器验证

1. 打开 Chrome/Firefox
2. 按 `F12` 打开开发者工具
3. 切换到 **Network** 标签
4. 访问：`https://perlou.top`
5. 点击任意图片请求
6. 查看 **Response Headers**

**期望看到**：

```
server: cloudflare
cf-cache-status: HIT
cf-ray: 8xxxxxxxxxxxxx-XXX
```

---

## 📊 步骤 9：查看分析数据

1. 点击 **Analytics & Logs** → **Traffic**
2. 你可以看到：
   - 📈 请求总量
   - 💾 带宽使用和节省
   - 🎯 缓存命中率
   - 🌍 访客地理分布
   - 🛡️ 威胁拦截统计

**目标指标**：

- 缓存命中率 > 80%
- 带宽节省 > 50%

---

## 🎯 完成检查清单

配置完成后，确认以下所有项：

**Cloudflare 配置**：

- [ ] 已注册 Cloudflare 账号
- [ ] 已添加域名 perlou.top
- [ ] DNS 已切换到 Cloudflare
- [ ] 收到 Cloudflare 激活邮件
- [ ] 网站状态显示 **Active**
- [ ] SSL/TLS 模式：**Full (strict)**
- [ ] **Always Use HTTPS**：已启用
- [ ] **Auto Minify**：已启用（JS/CSS/HTML）
- [ ] **Brotli**：已启用
- [ ] **Page Rules**：已创建图片缓存规则
- [ ] **HTTP/3**：已启用

**验证测试**：

- [ ] 响应头显示 `server: cloudflare`
- [ ] 图片请求显示 `cf-cache-status: HIT`
- [ ] 测试脚本运行成功
- [ ] 网站访问正常，速度明显提升

---

## 🔄 后续维护

### 清除缓存

更新内容后需要清除缓存：

1. 进入 **Caching** → **Configuration**
2. 选择：
   - **Purge Everything**：清除所有缓存
   - **Custom Purge**：清除特定 URL

### 开发模式

需要频繁测试时：

1. 进入 **Caching** → **Configuration**
2. 开启 **Development Mode**
3. 效果：暂停缓存 3 小时（自动恢复）

---

## ❓ 常见问题

### Q1: DNS 修改后多久生效？

A: 通常 5-30 分钟，最长可能 24-48 小时。可以用 `nslookup perlou.top` 检查。

### Q2: 缓存一直显示 MISS？

A: 检查:

1. Page Rules 是否创建成功
2. URL Pattern 是否正确
3. 访问的 URL 是否匹配规则

### Q3: 网站变慢了？

A: 可能原因：

1. DNS 还在生效中（等待）
2. 第一次访问（缓存未命中，正常）
3. 开启了 Rocket Loader（关闭试试）

### Q4: 如何回退？

A: 随时可以回退：

1. 登录域名注册商
2. 将 DNS 改回原来的服务器
3. 等待生效即可

---

## 🎉 恭喜！

完成以上步骤后，你的博客已经拥有：

- ✅ 全球 CDN 加速
- ✅ 自动 HTTPS
- ✅ DDoS 防护
- ✅ 智能缓存
- ✅ 流量分析

你的博客现在应该：

- 🚀 加载速度提升 80%+
- 🌍 全球访问体验一致
- 💰 节省 50%+ 带宽

享受企业级的 CDN 服务吧！🎊

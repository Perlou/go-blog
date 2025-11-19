# Cloudflare CDN 加速配置指南

## 🎯 目标

通过 Cloudflare 免费 CDN 加速博客图片和静态资源，提升全球访问速度。

---

## 📋 准备工作

- ✅ 域名：perlou.top（已有）
- ✅ 服务器：阿里云（已有）
- ⏳ Cloudflare 账号（需注册）

---

## 🚀 步骤一：注册 Cloudflare

### 1. 访问注册页面

打开：https://dash.cloudflare.com/sign-up

### 2. 填写注册信息

- 邮箱：你的常用邮箱
- 密码：设置强密码

### 3. 验证邮箱

点击邮件中的验证链接

---

## 🌐 步骤二：添加网站

### 1. 登录 Cloudflare 控制台

https://dash.cloudflare.com/

### 2. 添加站点

1. 点击 **"添加站点"** 或 **"Add a Site"**
2. 输入域名：`perlou.top`
3. 点击 **"添加站点"**

### 3. 选择计划

选择 **Free 免费计划**（$0/月）

### 4. 扫描 DNS 记录

- Cloudflare 会自动扫描现有 DNS 记录
- 检查扫描结果是否正确
- 确保有以下记录：

```
Type: A
Name: @
Content: [你的服务器IP]

Type: A
Name: www
Content: [你的服务器IP]
```

---

## 🔧 步骤三：修改域名 DNS

### 1. 获取 Cloudflare DNS 服务器

Cloudflare 会显示两个 DNS 服务器，类似：

```
ns1.cloudflare.com
ns2.cloudflare.com
```

**记下这两个地址！**

### 2. 去域名注册商修改 DNS

#### 如果是阿里云：

1. 登录 **阿里云控制台**
2. 进入 **域名** → **域名列表**
3. 找到 `perlou.top`，点击 **管理**
4. 点击 **DNS 修改**
5. 选择 **修改 DNS 服务器**
6. 输入 Cloudflare 的两个 DNS：
   - DNS 服务器 1：`ns1.cloudflare.com`
   - DNS 服务器 2：`ns2.cloudflare.com`
7. 点击 **确定**

#### 如果是腾讯云：

1. 登录 **腾讯云控制台**
2. 进入 **域名注册** → **我的域名**
3. 点击域名后的 **管理**
4. 找到 **DNS 服务器**，点击 **修改**
5. 输入 Cloudflare DNS
6. 保存

### 3. 等待 DNS 生效

- ⏱️ 通常需要 **5-30 分钟**
- 有时可能需要 **24-48 小时**
- Cloudflare 会发邮件通知激活成功

---

## ⚙️ 步骤四：优化配置（重要！）

### A. 速度优化

#### 1. 自动压缩

路径：**Speed** → **Optimization**

配置：

- ✅ **Auto Minify**
  - ☑️ JavaScript
  - ☑️ CSS
  - ☑️ HTML
- ✅ **Brotli**（更好的压缩算法）

#### 2. Rocket Loader（可选）

- ⚠️ 可能影响某些 JS，先不开启
- 如果网站正常可以尝试

---

### B. 图片优化（核心！）

路径：**Speed** → **Optimization**

配置：

- ✅ **Polish**：选择 **"Lossless"**（无损压缩）
- ✅ **Mirage**（移动端图片优化）

> **Polish 说明**：
>
> - Lossless：无损压缩，保持最高质量
> - Lossy：有损压缩，体积更小但质量略降

---

### C. 缓存配置

#### 1. 基础缓存

路径：**Caching** → **Configuration**

配置：

- **Browser Cache TTL**：4 hours
- **Caching Level**：Standard

#### 2. 图片缓存规则（重要！）

路径：**Rules** → **Page Rules** → **Create Page Rule**

**规则 1：图片缓存**

```
If the URL matches: perlou.top/images/*

Then the settings are:
  - Cache Level: Cache Everything
  - Edge Cache TTL: 1 month
```

点击 **Save and Deploy**

**规则 2：静态资源缓存**

```
If the URL matches: perlou.top/*.{jpg,jpeg,png,gif,css,js,woff,woff2}

Then the settings are:
  - Cache Level: Cache Everything
  - Browser Cache TTL: 1 year
  - Edge Cache TTL: 1 month
```

> 💡 **免费计划限制**：最多 3 条 Page Rules

---

### D. 安全配置

#### 1. SSL/TLS

路径：**SSL/TLS** → **Overview**

配置：

- 加密模式：选择 **"Full"** 或 **"Full (strict)"**

#### 2. Always Use HTTPS

路径：**SSL/TLS** → **Edge Certificates**

配置：

- ✅ **Always Use HTTPS**（强制 HTTPS）
- ✅ **Automatic HTTPS Rewrites**

---

## 🧪 步骤五：验证测试

### 1. DNS 检查

访问：https://www.whatsmydns.net/

- 输入：`perlou.top`
- 检查全球解析是否指向 Cloudflare

### 2. CDN 测试

打开浏览器开发者工具（F12）→ Network

访问：https://perlou.top/images/covers/langchain.jpg

查看响应头：

```
cf-cache-status: HIT  ← 表示命中 CDN 缓存
server: cloudflare   ← 表示走 Cloudflare
```

### 3. 速度测试

访问：https://tools.pingdom.com/

- 输入：`https://perlou.top`
- 测试加载速度

---

## 📊 预期效果

### 优化前

- 🇨🇳 国内访问图片：2-5 秒
- 🌏 海外访问：10+秒
- 📦 资源大小：原始大小

### 优化后

- 🇨🇳 国内访问：0.3-0.8 秒（提升 80%）
- 🌏 海外访问：0.5-1 秒（提升 90%）
- 📦 资源大小：减少 20-30%
- ⚡ 重复访问：几乎瞬间

---

## ⚠️ 注意事项

### 1. 开发模式

配置完成后，如果需要测试：

- 路径：**Caching** → **Configuration**
- 开启 **Development Mode**（暂停缓存 3 小时）

### 2. 清除缓存

更新内容后：

- 路径：**Caching** → **Configuration**
- 点击 **Purge Everything**（清除所有缓存）
- 或选择 **Custom Purge**（清除特定文件）

### 3. 服务器配置

保持你的 Nginx 配置不变，Cloudflare 会自动：

- 读取你的缓存头
- 应用优化策略
- 分发到全球节点

---

## 🎁 额外优势

### Cloudflare 免费功能

- ✅ DDoS 防护
- ✅ Web 应用防火墙（WAF）
- ✅ 流量分析
- ✅ 自动 HTTPS
- ✅ 全球 CDN（200+ 节点）
- ✅ 无限流量

### Analytics

路径：**Analytics & Logs** → **Web Analytics**

可以查看：

- 访问量统计
- 地理分布
- 流量趋势
- 热门内容

---

## 🆘 常见问题

### Q1: DNS 修改后多久生效？

A: 通常 5-30 分钟，最长可能 24-48 小时

### Q2: 会影响网站访问吗？

A: 不会，DNS 生效期间网站正常访问

### Q3: 免费版够用吗？

A: 对个人博客完全够用，无需升级

### Q4: 可以随时取消吗？

A: 可以，只需把 DNS 改回原来的即可

### Q5: 影响 Google Analytics 吗？

A: 不影响，统计正常工作

---

## 📞 需要帮助？

如果遇到问题：

1. 查看 Cloudflare 文档：https://developers.cloudflare.com/
2. 社区论坛：https://community.cloudflare.com/
3. 联系我（通过 About 页面）

---

## ✅ 配置清单

完成后检查：

- [ ] 注册 Cloudflare 账号
- [ ] 添加域名 perlou.top
- [ ] 修改 DNS 服务器
- [ ] 等待 DNS 生效（收到激活邮件）
- [ ] 配置 Auto Minify
- [ ] 配置 Polish（图片优化）
- [ ] 设置 Page Rules（缓存规则）
- [ ] 启用 Always Use HTTPS
- [ ] 验证 CDN 是否生效
- [ ] 测试网站速度

---

🎉 **配置完成后，你的博客将拥有企业级的全球 CDN 加速！**

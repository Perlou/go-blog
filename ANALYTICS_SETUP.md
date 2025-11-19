# Google Analytics 接入指南

## 快速接入（5 分钟）

### 1. 创建 Google Analytics 账号

1. 访问：https://analytics.google.com
2. 点击"开始衡量"
3. 创建账号和资源
   - **账号名称**：Perlou Blog
   - **资源名称**：perlou.top
   - **时区**：中国标准时间 GMT+8
   - **币种**：人民币

### 2. 获取衡量 ID

创建完成后，会得到一个 **衡量 ID**，格式为：

```
G-XXXXXXXXXX
```

### 3. 添加到 Hugo 配置

编辑 `hugo.yaml`，在顶部添加：

```yaml
services:
  googleAnalytics:
    id: "G-XXXXXXXXXX" # 替换为你的真实 ID
```

**完整示例**：

```yaml
baseurl: https://perlou.top/
languageCode: zh-cn
theme: hugo-theme-stack
paginate: 5
title: Perlou

# Google Analytics
services:
  googleAnalytics:
    id: "G-XXXXXXXXXX"
# 其他配置...
```

### 4. 验证配置

1. 本地构建：

   ```bash
   hugo && grep -r "G-XXXXXXXXXX" public/
   ```

2. 部署到服务器
3. 访问网站，打开浏览器开发者工具 → Network
4. 搜索 `google-analytics` 或 `gtag`
5. 应该能看到 GA 请求

### 5. 查看数据

- 访问：https://analytics.google.com
- 选择你的资源
- 等待 24-48 小时后会有初步数据

---

## 替代方案：Umami（自托管）

如果你更注重隐私，可以使用 Umami：

### 优点

- 完全自托管，数据掌握在自己手中
- 界面简洁美观
- 无需 cookie 同意
- 非常轻量

### 部署方式

1. 使用 Docker Compose 部署
2. 配置 PostgreSQL 数据库
3. 获取跟踪代码
4. 添加到 Hugo 模板

详细教程：https://umami.is/docs/getting-started

---

## 推荐建议

**初期推荐**：Google Analytics

- 免费、功能强大
- 无需额外服务器
- 数据分析能力强

**长期推荐**：Umami

- 隐私友好
- 数据掌控
- 界面简洁

---

## 要不要现在配置？

我可以帮你：

1. **如果你有 Google Analytics ID**：现在就添加到配置
2. **如果没有**：你可以自己创建后告诉我 ID
3. **想用 Umami**：我可以帮你准备 Docker Compose 配置

你想现在配置吗？还是稍后自己处理？

+++
date = '2025-11-21T11:58:20+08:00'
draft = false
title = '基于 Google Antigravity 开发 Httping：一个轻量级 API 测试工具的诞生'
image = '/images/covers/httping-project.svg'
categories = ['技术分享']
tags = ['开源项目', 'React', 'TypeScript', 'API测试', 'Cloudflare']
+++

这几天我基于 Google 的 Antigravity AI 助手，开发了一个名为 [Httping](https://perlou-httping.pages.dev/) 的轻量级 API 测试工具。这个项目从零到一的开发过程让我深刻体会到了 AI 辅助编程的强大之处，今天想和大家分享一下这段开发历程。

## 项目初衷

作为开发者，我们日常工作中经常需要测试各种 API 接口。虽然 Postman 这类工具功能强大，但有时候我们只是需要一个简单、快速、开箱即用的工具来快速验证接口。基于这个需求，我想开发一个：

- 🚀 **轻量级**：无需安装，打开浏览器即可使用
- ⚡ **快速**：界面简洁，响应迅速
- 🎨 **美观**：采用 Material Design 3，视觉体验舒适
- 💰 **零成本**：部署在 Cloudflare Pages，完全免费

## 技术选型

对于这样一个前端项目，我选择了以下技术栈：

- **React 19 + TypeScript 5.8**：类型安全 + 最新特性
- **Vite 7**：极速的开发体验
- **Tailwind CSS 3**：快速构建 UI
- **Zustand**：轻量级状态管理
- **Axios**：HTTP 请求处理
- **Cloudflare Pages**：零成本部署

这套技术栈的选择既保证了开发效率，又确保了最终产品的性能和用户体验。

## 开发历程

### 第一阶段：MVP 快速搭建

最初我向 Antigravity 描述了我的需求，AI 助手迅速帮我搭建起了项目的基础架构。包括：

- 基础的请求面板（支持 GET、POST、PUT、DELETE 等方法）
- 响应展示区域（Headers、Body、状态码）
- 简洁的界面布局

这个阶段大概只用了几个小时，一个可用的 MVP 就诞生了。

### 第二阶段：Material Design 重构

虽然基础功能已经实现，但初始的界面还比较简陋。我决定采用 Material Design 3 重构整个 UI。这个过程包括：

- 更新 Tailwind 配色方案，采用 Material 3 色彩系统
- 重构所有组件，使用 Material Design 的视觉语言
- 添加微动画和交互反馈
- 实现 Navigation Drawer 风格的历史记录侧边栏

重构后的界面焕然一新，专业度大幅提升。

### 第三阶段：核心功能完善

接下来是功能的迭代完善：

#### 1. **历史管理**

- 实现请求历史记录（最多 20 条）
- 支持单条删除
- 快速复用历史请求

#### 2. **环境变量支持**

- 开发环境 / 生产环境切换
- URL 中使用 `{{变量名}}` 语法
- 动态替换请求参数

#### 3. **认证支持**

- Bearer Token 认证
- Basic Auth 认证
- 灵活的认证配置

#### 4. **cURL 导入**

这是一个非常实用的功能。开发过程中遇到了一些挑战：

最初我考虑使用 `curlconverter` 库，但发现它会显著增加打包体积。于是我和 Antigravity 一起，实现了一个轻量级的自定义 cURL 解析器，既满足了功能需求，又保持了项目的轻量特性。

### 第四阶段：质量保障

为了确保代码质量，我们：

- 使用 Vitest + React Testing Library 编写单元测试
- 测试核心工具函数和关键组件
- 确保所有功能都有测试覆盖

### 第五阶段：部署与发布

最后是部署环节：

- 配置 Cloudflare Pages 部署
- 设置 GitHub Actions 自动发布流程
- 域名绑定：[perlou-httping.pages.dev](https://perlou-httping.pages.dev/)

整个部署过程非常顺畅，Cloudflare Pages 的速度和稳定性都很出色。

## 核心功能展示

### 快捷键

为了提升效率，我内置了几个实用的快捷键：

- `Ctrl/Cmd + Enter` - 发送请求
- `Ctrl/Cmd + K` - 聚焦 URL 输入框
- `?` - 显示帮助

### 智能 URL 处理

如果输入的 URL 没有协议前缀，工具会自动添加 `http://`，减少手动输入的麻烦。

### 响应详情

响应面板清晰展示：

- HTTP 状态码和响应时间
- 完整的 Response Headers
- 格式化的响应 Body
- 支持 JSON、文本等多种内容类型

## 与 AI 协作的感受

这次开发过程中，Antigravity 扮演了一个非常称职的"编程搭档"角色：

### 优势

1. **快速原型**：从想法到可运行的原型，速度极快
2. **最佳实践**：AI 会主动建议更好的实现方案
3. **代码质量**：自动遵循代码规范，类型安全
4. **测试编写**：自动生成测试用例，覆盖主要场景

### 挑战

1. **需求明确**：需要清晰地描述需求，避免歧义
2. **架构把控**：整体架构和设计方向需要人工把控
3. **细节调整**：UI 细节和用户体验需要反复迭代

## 未来规划

Httping 还有很多可以改进的地方：

- [ ] 请求集合管理（Collections）
- [ ] 导入/导出功能
- [ ] GraphQL 支持
- [ ] WebSocket 测试
- [ ] 更多认证方式（OAuth 等）
- [ ] 响应数据可视化

## 总结

这次开发经历让我深刻感受到：

1. **AI 辅助开发已经非常成熟**：从代码生成到测试编写，AI 都能提供实质性帮助
2. **人机协作是最优方案**：AI 负责执行，人负责创意和把控
3. **快速迭代很重要**：MVP → 完善功能 → 优化体验，步步为营
4. **开源很有意义**：把工具分享出来，让更多人受益

如果你对 Httping 感兴趣，欢迎访问：

- 🌐 **在线体验**：[perlou-httping.pages.dev](https://perlou-httping.pages.dev/)
- 💻 **源码仓库**：[github.com/Perlou/httping](https://github.com/Perlou/httping)

如果觉得有帮助，欢迎给项目一个 Star ⭐，也欢迎提 Issue 和 PR！

---

_本文记录于 2025 年 11 月 21 日，Httping 的开发还在继续..._

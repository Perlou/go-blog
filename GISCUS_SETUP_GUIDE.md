# Giscus 评论系统配置

## 前置要求

- 仓库必须是公开的（Public）
- 已启用 GitHub Discussions

## 配置步骤

### 1. 启用 Discussions

1. 打开仓库：https://github.com/Perlou/go-blog
2. `Settings` → `General` → `Features`
3. 勾选 ✅ `Discussions`

### 2. 安装 Giscus App

1. 访问：https://github.com/apps/giscus
2. 点击 `Install`
3. 选择仓库：`Perlou/go-blog`

### 3. 获取配置参数

1. 访问：https://giscus.app/zh-CN
2. 填写仓库：`Perlou/go-blog`
3. 选择 Discussion 分类：`Announcements`
4. 页面映射：`pathname`
5. 主题：`preferred_color_scheme`

### 4. 复制配置参数

从生成的代码中获取：

```
data-repo-id: R_kgDOxxxxxxx
data-category: Announcements
data-category-id: DIC_kwDOxxxxxxx
```

### 5. 添加到配置

将以上参数添加到 `hugo.yaml` 的 Giscus 配置中。

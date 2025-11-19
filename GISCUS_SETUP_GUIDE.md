# Giscus 评论系统配置指南

## 步骤 1: 启用 GitHub Discussions

1. **打开你的仓库**
   访问：https://github.com/Perlou/go-blog

2. **进入设置**
   点击仓库顶部的 `Settings` 标签

3. **启用 Discussions**
   - 在左侧菜单找到 `General`
   - 向下滚动到 `Features` 部分
   - 勾选 ✅ `Discussions`
   - 点击 `Save changes`

## 步骤 2: 安装 Giscus App

1. **访问 Giscus App 页面**
   https://github.com/apps/giscus

2. **安装应用**
   - 点击绿色的 `Install` 按钮
   - 选择 `Only select repositories`
   - 从下拉菜单选择 `Perlou/go-blog`
   - 点击 `Install`

## 步骤 3: 获取 Giscus 配置参数

1. **访问 Giscus 配置页面**
   https://giscus.app/zh-CN

2. **填写配置信息**

   **仓库**：

   ```
   Perlou/go-blog
   ```

   填写后，页面会自动验证仓库是否满足条件（应显示绿色 ✅）

3. **选择 Discussion 分类**

   - 在 "Discussion 分类" 下拉菜单中选择 `Announcements`
   - 或者点击 "创建新分类" 链接，在 GitHub 创建一个名为 "Comments" 的分类

4. **页面 ↔️ Discussion 映射关系**
   选择：`Discussion 的标题包含页面的 pathname`

5. **特性**

   - ✅ 启用主评论框上方的反应
   - ✅ 懒加载评论

6. **主题**
   选择：`preferred_color_scheme`（自动跟随网站主题）

7. **获取配置代码**

   滚动到页面底部的 "启用 giscus" 部分，你会看到类似这样的代码：

   ```html
   <script src="https://giscus.app/client.js"
        data-repo="Perlou/go-blog"
        data-repo-id="R_kgDOxxxxxxx"
        data-category="Announcements"
        data-category-id="DIC_kwDOxxxxxxx"
        ...
   ```

   **请复制以下四个参数的值**：

   - `data-repo-id` 的值（类似 `R_kgDOxxxxxxx`）
   - `data-category` 的值（如 `Announcements`）
   - `data-category-id` 的值（类似 `DIC_kwDOxxxxxxx`）
   - `data-mapping` 的值（应该是 `pathname`）

## 步骤 4: 提供参数给我

完成上述步骤后，请把以下信息告诉我：

```
repo-id: [你的 repo ID]
category: [你的 category 名称]
category-id: [你的 category ID]
```

例如：

```
repo-id: R_kgDOL1234567
category: Announcements
category-id: DIC_kwDOL1234567Y
```

我会用这些参数配置 `hugo.yaml`，然后就可以测试和部署了！

---

## 💡 提示

- 如果看不到 Discussions 选项，可能是私有仓库。确保仓库是公开的（Public）
- Giscus App 的安装是免费的，无需担心
- 配置完成后，所有评论会存储在 GitHub Discussions 中，完全免费且无广告

---
name: hugo-stack-overrides
description: Use when customizing a Hugo Theme Stack (or any Hugo theme installed as git submodule) without modifying theme source code. Covers the layouts/ override pattern, partials customization, performance optimizations (responsive images, preload, lazy loading), and theme upgrade safety. Use when the user says "改样式 / 调布局 / 主题升级 / 优化 LCP / 改主题但别破坏升级".
version: 1.0.0
license: MIT
---

# Hugo Theme Stack Customization · 不破坏升级的主题改造

> 适用于 Hugo + 任何作为 git submodule 引入的主题（特别是 [Stack](https://github.com/CaiJimmy/hugo-theme-stack)）。
> 把"改主题"从"破坏未来升级路径"改成"override 安全路径"。

## 何时调用本 skill

**应当**：

- 用户说"改主题样式 / 调布局"
- 用户要做性能优化（LCP / FCP / 图片懒加载）
- 主题需要 hook（评论懒加载、自定义 head 标签、SEO 增强）
- 主题升级前 / 升级中 / 升级后排错
- 在主题里加自己的 partial（不在主题原生 list 里的）

**不应**：

- 用户的主题不是 git submodule（直接改源码反而简单）
- 用户要做主题级的全局变更（应 fork 主题）
- 主题不支持 layouts/ override（极少；Stack 完全支持）

---

## 核心原则

### 一句话

**所有自定义放 `layouts/`，禁止动 `themes/<theme>/` 下任何文件。**

### 为什么

| 改主题源码 | 走 layouts/ override |
|---|---|
| 主题升级覆盖你的修改 | 升级安全 |
| `git submodule update` 报冲突 | 主题作为 submodule 干净更新 |
| 修改无 git history（在 submodule 内） | 修改进主仓库 git history |
| 别人 fork 你的仓库还要重新改 | clone 即用 |

### 判断标准

**`themes/` 下任何文件被修改 = 反模式。**

可以这样验证：

```bash
git diff --stat themes/    # 应该完全空
cd themes/hugo-theme-stack && git status    # 应该完全空
```

---

## 覆盖优先级

Hugo 查找 layout 文件时，优先级如下（高到低）：

```
1. layouts/<path>.html              （项目自定义）
2. themes/<theme>/layouts/<path>.html  （主题默认）
3. Hugo 内置 fallback
```

也就是说：**项目 layouts/ 下的文件路径与主题里完全相同 → 自动覆盖。**

---

## 标准改造流程

### 1. 找到主题里要改的文件

```bash
# 例：想改首页文章列表的样式
find themes/hugo-theme-stack/layouts -name "default*.html" | grep article-list
# → themes/hugo-theme-stack/layouts/partials/article-list/default.html
```

### 2. 复制到项目 layouts/，路径完全相同

```bash
mkdir -p layouts/partials/article-list
cp themes/hugo-theme-stack/layouts/partials/article-list/default.html \
   layouts/partials/article-list/default.html
```

### 3. 在副本上改

```html
{{/* 原始：主题默认 article-list */}}
{{/* 修改原因：性能优化 —— 用 helper/image.html 输出 srcset */}}

<article class="article-list">
  ...
  {{/* 把原来的图片标签替换为响应式 srcset */}}
  {{ partial "helper/image.html" . }}
  ...
</article>
```

**总是在文件顶部加注释**说明：
- 这是哪个主题文件的 override
- 改了什么
- 为什么改

### 4. 本地验证

```bash
hugo server                  # 看页面是否还正常渲染
hugo --cleanDestinationDir   # 清缓存重建一次
```

### 5. commit

```bash
git add layouts/partials/article-list/default.html
git commit -m "style: optimize article-list with responsive image srcset"
```

---

## 5 个常用 override 模式

### 1. 响应式图片处理（封面图自动 srcset）

`layouts/partials/helper/image.html`：

```go-html-template
{{/* 用法: {{ partial "helper/image.html" . }} */}}
{{/* 自动生成 800w + 1600w 两种尺寸的 webp，输出 srcset */}}

{{ if .Params.image }}
  {{ $img := resources.Get .Params.image }}
  {{ if $img }}
    {{ $w800 := $img.Resize "800x webp q80" }}
    {{ $w1600 := $img.Resize "1600x webp q80" }}
    <img
      src="{{ $w800.RelPermalink }}"
      srcset="{{ $w800.RelPermalink }} 800w, {{ $w1600.RelPermalink }} 1600w"
      sizes="(max-width: 800px) 100vw, 800px"
      alt="{{ .Title }}"
      loading="lazy"
      decoding="async">
  {{ end }}
}}
```

### 2. 资源预连接（`head/custom.html`）

`layouts/partials/head/custom.html`：

```html
{{/* DNS prefetch + preconnect 第三方资源（Giscus / GA 等） */}}
<link rel="dns-prefetch" href="https://giscus.app">
<link rel="dns-prefetch" href="https://www.googletagmanager.com">
<link rel="preconnect" href="https://giscus.app" crossorigin>

{{/* 预加载头像（首页 LCP 优化） */}}
<link rel="preload" as="image"
      href="{{ .Site.Params.sidebar.avatar.src | absURL }}"
      fetchpriority="high">
```

### 3. CSS preload（`head/style.html`）

```html
{{/* 主样式 preload 加速 FCP */}}
{{ $style := resources.Get "scss/main.scss" | resources.ToCSS | resources.Minify }}
<link rel="preload" as="style" href="{{ $style.RelPermalink }}">
<link rel="stylesheet" href="{{ $style.RelPermalink }}">
```

### 4. 评论懒加载（`comments/provider/giscus.html`）

```html
{{/* IntersectionObserver 延迟加载，省 ~200KB JS 在首屏 */}}
<div id="giscus-container" data-giscus-loaded="false"></div>
<script>
  const obs = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && document.getElementById('giscus-container').dataset.giscusLoaded === 'false') {
      document.getElementById('giscus-container').dataset.giscusLoaded = 'true';
      const s = document.createElement('script');
      s.src = 'https://giscus.app/client.js';
      s.setAttribute('data-repo', '{{ .Site.Params.giscus.repo }}');
      // ...其它 data-* 配置
      s.crossOrigin = 'anonymous';
      s.async = true;
      document.getElementById('giscus-container').appendChild(s);
    }
  }, { rootMargin: '200px' });
  obs.observe(document.getElementById('giscus-container'));
</script>
```

### 5. 自定义 footer / GA 接入

```html
{{/* layouts/partials/google_analytics.html */}}
{{ if .Site.Services.GoogleAnalytics.ID }}
<script async src="https://www.googletagmanager.com/gtag/js?id={{ .Site.Services.GoogleAnalytics.ID }}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', '{{ .Site.Services.GoogleAnalytics.ID }}');
</script>
{{ end }}
```

---

## 主题升级流程

```bash
# 1. 看升级会带来什么
cd themes/hugo-theme-stack
git fetch origin
git log HEAD..origin/master --oneline

# 2. 升级
cd ../..
git submodule update --remote --merge themes/hugo-theme-stack

# 3. 检查所有自定义还在不在
hugo server
# 浏览器逐页核对：首页 / 文章页 / 归档页 / 关于页 / 评论 / 搜索

# 4. 如果某个 override 文件路径主题里被重命名 / 删除
#    主题升级日志会提示；调整 layouts/ 下对应路径
ls layouts/partials/<old-path>.html        # 检查老 override
find themes/hugo-theme-stack/layouts -name "<old-name>*"   # 看主题里新的位置

# 5. 确认无误后 commit
git add themes/hugo-theme-stack
git commit -m "chore: bump theme to vX.Y.Z"
```

---

## 反模式（红旗）

| 反模式 | 怎么发现 | 修正 |
|---|---|---|
| 直接改 `themes/` 下文件 | `git diff themes/` 非空 | 把 diff 内容挪到 `layouts/` 同路径，撤销 themes/ 修改 |
| `layouts/` 路径与主题不一致 | override 不生效 | 路径必须**完全相同**（含 partials/ 子目录） |
| override 没注释 | 后人不知改了什么 | 顶部 `{{/* ... 改动原因 ... */}}` |
| 主题升级后没跑 hugo server | 部署上线才发现挂 | 升级流程必跑本地预览 |
| 把整个主题 fork | 维护负担巨大 | 90% 的需求 layouts/ override 就够了 |
| 改主题里的 i18n | 翻译跟着升级丢失 | 走 `i18n/<lang>.toml` 文件覆盖 |
| 改主题 SCSS 源码 | 升级带新样式时冲突 | 在 `assets/scss/custom.scss` 里加 override |

---

## 当 layouts/ override 不够用时

少数情况主题没暴露你想 hook 的位置。三种回退：

### 1. 全局注入：`baseof.html` 整体替换

```bash
cp themes/hugo-theme-stack/layouts/_default/baseof.html layouts/_default/baseof.html
# 在副本里加你的全局 hook
```

### 2. 用 `partials` 桥接

把主题没有的 partial 加到 `layouts/partials/`，然后在 baseof override 里
`{{ partial "<your-partial>" . }}` 调用。

### 3. fork 主题

仅在大改的情况下。fork 后把 git submodule URL 改为你 fork 的仓库。代价：
要自己跟主题更新。

---

## 工作流程（Claude 应用本 skill 时）

当用户说"改主题样式 / 改布局 / 调主题"时：

1. **检查主题是否被脏改**：`git diff --stat themes/`，非空就先恢复
2. **找到主题里的目标文件**：`find themes/<theme>/layouts -name "<keyword>*"`
3. **复制到 `layouts/` 同路径**（含完整子目录）
4. **改副本，顶部加注释**说明改动原因
5. **`hugo server` 验证**改动生效
6. **commit**：`style:` / `feat:` 前缀

如果用户问"主题升级怎么办"：

1. 跑标准升级流程（看上面）
2. 提醒**逐页验证**所有 override 仍然生效
3. 升级 commit 必须独立，不要混杂功能改动

---

## 一句话精要

**主题作为 git submodule 的核心契约是"主题源码不可变"。** 守住这条，
你的所有定制就永远跟着主题升级走，不会某天因主题更新而集体失效。
`layouts/` 是你和主题之间的安全接口。

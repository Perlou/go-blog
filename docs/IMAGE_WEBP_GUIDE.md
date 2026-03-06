# 图片 WebP 转换指南

本项目所有封面图使用 WebP 格式，存放在 `assets/images/bg/` 目录下。Hugo 构建时会自动生成 800w/1600w 响应式缩略图。

## 环境准备

```bash
# macOS
brew install webp

# Ubuntu/Debian
sudo apt install webp

# 验证安装
cwebp -version
```

## 单张转换

```bash
cwebp -q 80 input.jpg -o output.webp
```

参数说明：

| 参数          | 说明                                      |
| ------------- | ----------------------------------------- |
| `-q 80`       | 质量 0-100，80 为推荐值（体积与画质平衡） |
| `-resize W H` | 调整尺寸，H 设为 0 表示按比例缩放         |
| `-mt`         | 多线程加速                                |

## 批量转换

将所有 JPG 转为 WebP 并放入 `assets/images/bg/`：

```bash
for f in *.jpg; do
  cwebp -q 80 "$f" -o "assets/images/bg/${f%.jpg}.webp"
done
```

## 在文章中使用

1. 将 WebP 图片放入 `assets/images/bg/`
2. 在文章 front matter 中引用：

```toml
+++
title = '文章标题'
image = '/images/bg/your-image.webp'
+++
```

Hugo 会通过自定义的 `helper/image.html` 自动：

- 从 `assets/` 目录查找图片
- 生成 **800w** 和 **1600w** 两种尺寸的 srcset
- 文章列表页加载 800px 缩略图，Retina 屏加载 1600px

## 推荐图片规格

| 项目     | 建议值                 |
| -------- | ---------------------- |
| 格式     | WebP                   |
| 原始尺寸 | 1920×1920 或 1920×1080 |
| 质量     | 80                     |
| 单张大小 | < 300KB                |

> **提示**：原始图片尺寸建议 ≥ 1600px 宽，这样 Hugo 生成的 1600w 版本不会被拉伸。小于 1600px 的图片 Hugo 会直接使用原图。

## 从其他格式转换

```bash
# PNG → WebP
cwebp -q 80 input.png -o output.webp

# 带透明通道的 PNG（保留 alpha）
cwebp -q 80 -alpha_q 90 input.png -o output.webp

# 调整尺寸（宽度 1920，高度按比例）
cwebp -q 80 -resize 1920 0 input.jpg -o output.webp
```

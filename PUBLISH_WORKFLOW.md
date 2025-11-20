# 发布工作流

## 快速发布

使用发布脚本自动提交并触发部署：

```bash
./publish.sh
```

脚本会自动：

1. 显示文件改动
2. 提示输入提交信息
3. 推送到 GitHub
4. 触发 GitHub Actions 部署（需安装 `gh` CLI）

## GitHub CLI 安装

macOS:

```bash
brew install gh
gh auth login
```

## 手动触发部署

如果没有安装 `gh` CLI，访问：

```
https://github.com/Perlou/go-blog/actions/workflows/deploy.yml
```

点击 `Run workflow` 按钮。

## 提交信息规范

- `post: 发布新文章`
- `feat: 添加新功能`
- `fix: 修复问题`
- `docs: 更新文档`

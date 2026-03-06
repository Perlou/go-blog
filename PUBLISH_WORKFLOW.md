# 发布工作流

## 快速发布

使用发布脚本自动提交并触发部署：

```bash
./publish.sh
```

脚本会自动：

1. 显示文件改动
2. 提示输入提交信息
3. 选择暂存模式（仅已跟踪文件 / 所有改动）
4. 二次确认待提交文件列表
5. 运行本地构建检查（`hugo --minify`）
6. 推送到 GitHub
7. 触发 GitHub Actions 部署（需安装 `gh` CLI）

## 暂存模式说明

- `1`（默认）：仅暂存已跟踪文件（`git add -u`），更安全
- `2`：暂存所有改动（`git add -A`），适合发布新文章等新增文件场景

如果选择 `1` 且只有新增文件，脚本会提示“暂存区为空”，重新运行并选择 `2` 即可。

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

## CI 构建检查

仓库新增了 `.github/workflows/ci.yml`，会在 `push` 和 `pull_request` 时自动执行 Hugo 构建，提前发现配置或模板问题。

## 提交信息规范

- `post: 发布新文章`
- `feat: 添加新功能`
- `fix: 修复问题`
- `docs: 更新文档`

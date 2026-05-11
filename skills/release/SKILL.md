---
name: release
description: Dify vLLM provider 完整 release 流程。当需要发布新版本、构建 .difypkg、打 tag、创建 GitHub release、更新 fork、提交 PR 到 langgenius/dify-plugins 时使用。触发词包括 "release"、"发布"、"构建版本"、"提交 PR"、"打包插件" 等。
allowed-tools: Bash(git:*), Bash(dify:*), Bash(gh:*)
---

# Release Workflow

完成 bug 修复/功能开发后，发布新版本到 Dify Marketplace 的完整流程。

## 1. Bump Version

编辑 `manifest.yaml`，将两处 `version` 字段自增：

```yaml
# 第1行
version: 0.2.2

# meta 段内
meta:
  version: 0.2.2
```

## 2. Build Plugin Package

```bash
dify plugin package . -o /tmp/vllm-{version}.difypkg
```

## 3. Commit & Tag

```bash
git add manifest.yaml models/llm/llm.py    # 及相关改动文件
git commit -m '<版本号: 简短标题>

## 问题/错误
<描述>

## 发现过程
<描述>

## 实现方案
<描述>

## 实现过程
<描述>

## 总结
<描述>'

git tag -a v{version} -m "v{version}: <描述>"
git push origin master
git push origin v{version}
```

## 4. Create GitHub Release

直接用 `gh release create`（不用 draft）：

```bash
gh release create v0.2.2 \
  --repo yangyaofei/dify-vllm-provider \
  --title "v0.2.2" \
  --notes '<release notes in markdown>' \
  /tmp/vllm-0.2.2.difypkg

# 如果 release 已存在需更新内容:
gh release edit v0.2.2 --repo yangyaofei/dify-vllm-provider --notes '<merged notes>'

# 上传/替换附件:
gh release upload v0.2.2 /tmp/vllm-0.2.2.difypkg \
  --repo yangyaofei/dify-vllm-provider --clobber
```

## 5. Update Fork (yangyaofei/dify-plugins)

```bash
# 浅克隆（仓库很大，很多二进制文件）
git clone --depth 1 git@github.com:yangyaofei/dify-plugins.git /tmp/dify-plugins

# 删除旧版本 .difypkg，放入新版本
rm /tmp/dify-plugins/yangyaofei/dify-vllm-provider-v{old}.difypkg
cp /tmp/vllm-{version}.difypkg /tmp/dify-plugins/yangyaofei/dify-vllm-provider-v{version}.difypkg

# 提交 & push
cd /tmp/dify-plugins
git add yangyaofei/
git commit -m 'vllm provider v{version}: <描述>'
git push origin main
```

**注意**: commit message 中 issue 引用必须用跨仓库格式：

```
# 错误（会引用当前 repo 的 issue）:
修复 #34 思考模式标记格式

# 正确:
修复 yangyaofei/dify-vllm-provider#34 思考模式标记格式
```

## 6. Submit/Update PR to Official

如果已有旧版本 PR 打开，直接 push 同分支即可自动更新：

```bash
# 更新 PR 标题
gh pr edit <PR_NUMBER> --repo langgenius/dify-plugins --title "vllm provider v{version}"

# 更新 PR 描述（使用跨仓库 issue 引用）
gh pr edit <PR_NUMBER> --repo langgenius/dify-plugins --body '<Plugin Submission Form>'
```

新建 PR：

```bash
gh pr create \
  --repo langgenius/dify-plugins \
  --head yangyaofei:main \
  --base main \
  --title "vllm provider v0.2.2" \
  --body '<Plugin Submission Form>'
```

PR body 必须包含 [Plugin Submission Form](https://github.com/langgenius/dify-plugins) 要求的完整表单（6 个 section）。

## 7. Notify Related Issues

```bash
gh issue comment <ISSUE_NUMBER> \
  --repo yangyaofei/dify-vllm-provider \
  --body 'v{version} has been submitted via PR https://github.com/langgenius/dify-plugins/pull/<PR_NUMBER>.'
```

## 附: 检查清单

- [ ] manifest.yaml 版本号已自增
- [ ] `dify plugin package` 构建成功
- [ ] commit message 遵循规范格式
- [ ] git tag 已创建并推送
- [ ] GitHub release 创建成功且附件上传正确
- [ ] dify-plugins fork 已更新
- [ ] dify-plugins commit message 使用跨仓库 issue 引用
- [ ] PR 标题/描述已更新
- [ ] 相关 issue 已通知

---
name: release
description: Dify vLLM provider 完整 release 流程。当需要发布新版本、构建 .difypkg、打 tag、创建 GitHub release、更新 fork、提交 PR 到 langgenius/dify-plugins 时使用。触发词包括 "release"、"发布"、"构建版本"、"提交 PR"、"打包插件" 等。
allowed-tools: Bash(git:*), Bash(dify:*), Bash(gh:*), Bash(agent-browser:*)
---

# Release Workflow

完成 bug 修复/功能开发后，发布新版本到 Dify Marketplace 的完整流程。

## Tool Preference

- **Primary**: 所有 GitHub 操作优先使用 `gh` CLI（更快、更可靠）
- **Fallback**: 当 `gh` 不可用时，使用 `agent-browser` 操作 GitHub web UI

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

**Primary (`gh`)** — 直接 create，不用 draft：

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

**Fallback (`agent-browser`)**:

```bash
# 打开 new release 页面
agent-browser open https://github.com/yangyaofei/dify-vllm-provider/releases/new
agent-browser snapshot -i

# 选择 tag，填写 title/notes
agent-browser click @<tag-dropdown-ref>
agent-browser click @<v0.2.2-menuitemradio>
agent-browser fill @<title-textbox> "v0.2.2"
# 通过 eval 设置 release notes (textarea 中的引号/特殊字符容易导致 bash 错误)
cat <<'EOF' | agent-browser eval --stdin
const ta = document.querySelector('#fc-release_body');
if (ta) { ta.value = `notes content`; ta.dispatchEvent(new Event('input', {bubbles: true})); }
EOF

# 上传 .difypkg (用隐藏的 file input)
agent-browser upload "#releases-upload" "/tmp/vllm-0.2.2.difypkg"
# 或通过 eval 找 file input:
cat <<'EOF' | agent-browser eval --stdin
const inputs = document.querySelectorAll('input[type="file"]');
JSON.stringify(Array.from(inputs).map((el,i) => ({i, id: el.id, accept: el.accept})));
EOF

# 发布
agent-browser click @<publish-button>
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

**Primary (`gh`)**:

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

**Fallback (`agent-browser`)**:

```bash
# 从 yangyaofei/dify-plugins 发起 PR
agent-browser open https://github.com/yangyaofei/dify-plugins
# 点击 "Contribute" → "Open pull request"，选择 langgenius/dify-plugins:main 作为 base
# 填写 title 和 body，提交
```

## 7. Handle Review Feedback

**Primary (`gh`)**:

```bash
# 查看 reviews
gh pr view <PR_NUMBER> --repo langgenius/dify-plugins --json reviews --jq '.reviews[]'

# 查看 CI checks
gh pr checks <PR_NUMBER> --repo langgenius/dify-plugins
```

**Fallback (`agent-browser`)**:

```bash
agent-browser open https://github.com/langgenius/dify-plugins/pull/<PR_NUMBER>
agent-browser snapshot -i   # 查看 review comments 和 CI status
```

## 8. Notify Related Issues

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

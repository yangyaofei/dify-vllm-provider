# Release Workflow

完成 bug 修复/功能开发后的完整 release 流程。

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
# 例如:
dify plugin package . -o /tmp/vllm-0.2.2.difypkg
```

## 3. Commit & Tag

```bash
git add manifest.yaml models/llm/llm.py    # 及相关改动的文件
git commit -m 'v0.2.2: <简要描述>'
git tag -a v0.2.2 -m "v0.2.2: <描述>"
git push origin master
git push origin v0.2.2
```

**commit message 规范**:
```
<版本号: 简短标题>

## 问题/错误
<发现了什么问题或需求是什么>

## 发现过程
<如何发现问题的、根因分析>

## 实现方案
<准备如何实现、设计决策>

## 实现过程
<具体做了什么修改>

## 总结
<最终结论和注意事项>
```

## 4. Create GitHub Release

直接用 `gh` CLI，不要用 draft：

```bash
gh release create v0.2.2 \
  --repo yangyaofei/dify-vllm-provider \
  --title "v0.2.2" \
  --notes '<release notes in markdown>' \
  /tmp/vllm-0.2.2.difypkg

# 如果 release 已存在但需更新:
gh release edit v0.2.2 \
  --repo yangyaofei/dify-vllm-provider \
  --notes '<merged notes>'

# 上传/替换附件:
gh release upload v0.2.2 /tmp/vllm-0.2.2.difypkg \
  --repo yangyaofei/dify-vllm-provider --clobber
```

## 5. Update Dify Marketplace Fork

将构建好的 `.difypkg` 提交到 `yangyaofei/dify-plugins` fork：

```bash
# 浅克隆 (仓库很大)
git clone --depth 1 git@github.com:yangyaofei/dify-plugins.git /tmp/dify-plugins

# 删除旧版本，放入新版本
rm /tmp/dify-plugins/yangyaofei/dify-vllm-provider-v0.2.1.difypkg
cp /tmp/vllm-0.2.2.difypkg /tmp/dify-plugins/yangyaofei/dify-vllm-provider-v0.2.2.difypkg

# 提交 & push
cd /tmp/dify-plugins
git add yangyaofei/
git commit -m 'vllm provider v0.2.2: <描述>'
git push origin main
```

**注意**: commit message 中的 issue 引用必须用跨仓库格式，用裸 `#NN` 会错误引用当前 repo 的 issue：

```
# 错误 (会引用 yangyaofei/dify-plugins#34):
修复 #34 思考模式标记格式

# 正确:
修复 yangyaofei/dify-vllm-provider#34 思考模式标记格式
```

## 6. Submit PR to Official (langgenius/dify-plugins)

如果已有旧版本 PR 打开，直接 push 到同一分支即可自动更新 PR：

```bash
# 更新 PR 标题
gh pr edit <PR_NUMBER> --repo langgenius/dify-plugins --title "vllm provider v0.2.2"

# 更新 PR 描述 (也需要用跨仓库 issue 引用)
gh pr edit <PR_NUMBER> --repo langgenius/dify-plugins --body '<完整表单>'
```

如果还没有 PR，从 yangyaofei/dify-plugins 的 main 分支创建：

```bash
# 方法1: 浏览器打开
open https://github.com/yangyaofei/dify-plugins

# 方法2: gh CLI (需要 fork 同步后)
gh pr create \
  --repo langgenius/dify-plugins \
  --head yangyaofei:main \
  --base main \
  --title "vllm provider v0.2.2" \
  --body '<Plugin Submission Form>'
```

## 7. Notify Related Issues

在 `yangyaofei/dify-vllm-provider` 的 issue 中评论，引用已提交的 PR：

```bash
gh issue comment <ISSUE_NUMBER> \
  --repo yangyaofei/dify-vllm-provider \
  --body 'v0.2.2 has been submitted via PR https://github.com/langgenius/dify-plugins/pull/<PR_NUMBER>.'
```

## 附: 检查清单

- [ ] manifest.yaml 版本号已自增
- [ ] `dify plugin package` 构建成功
- [ ] commit message 遵循规范格式
- [ ] git tag 已创建并推送
- [ ] GitHub release 创建成功且附件上传正确
- [ ] dify-plugins fork 已更新 (.difypkg 替换)
- [ ] dify-plugins commit message 使用跨仓库 issue 引用
- [ ] PR 标题/描述已更新或已新建
- [ ] 相关 issue 已通知

# vLLM Dify Provider

Dify 自定义模型供应商，基于 vLLM 的 OpenAI-Compatible Server，支持 [extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters) 和思考模式。

基于 Dify 官方 OpenAI-API-compatible 插件扩展，专为 vLLM OpenAI-Compatible Server 提供 `extra_body` 和思考模式支持。

## 最新: v0.2.2

### 修复
- **#34**: 修复思考模式标记格式 — `<think/>`/`</think/>` → `<think>`/`</think>`，与 [dify-official-plugins](https://github.com/langgenius/dify-official-plugins) openai_api_compatible 官方插件 1:1 对齐
- **#31**: 修复 `extra_body` 参数传递 — JSON 内容从嵌套的 `"extra_body"` key 提升到请求体顶层，使 vLLM 非标准参数正确生效

### 功能 (自 v0.2.0 起)
- **extra_body**: 以 JSON 格式直接传递 vLLM extra parameters
- **思考模式**: `enable_thinking` 开关配合 `compatibility_mode`（strict/extended）
- **推理力度**: 支持 `reasoning_effort`（none/low/medium/high），vLLM 原生支持
- **兼容模式**: Extended 模式将 `chat_template_kwargs`、`thinking`、`enable_thinking` 注入顶层
- **结构化输出**: 支持 `response_format`、`json_schema`、`reasoning_format`
- **思考内容过滤**: 关闭 thinking 时自动过滤 `<think>...</think>` 内容
- **思考内容清理**: 请求前自动清理历史消息中的思考内容
- **vLLM Reasoning 字段**: 优先读取 `reasoning`（vLLM >= 0.17.1），回退至 `reasoning_content`

### v0.2.0 基线
> **重大变更**: 移除所有旧参数，统一使用 `extra_body` 传递所有额外参数。

## 使用

### 添加模型
与 OpenAI-API-compatible 相同，选择 "Vllm" 供应商：

![添加模型](./_assets/add_model.png)

### 配置 extra_body
在 `extra_body` JSON 文本框中传递额外参数：

![使用 guided](./_assets/use_guided.png)

示例：
```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

## 仓库
https://github.com/yangyaofei/dify-vllm-provider

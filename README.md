# vLLM Dify Provider

Dify custom model provider for vLLM's OpenAI-Compatible Server, supporting [extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters) and thinking mode features.

基于 Dify 官方 OpenAI-API-compatible 插件扩展，专为 vLLM OpenAI-Compatible Server 提供 `extra_body` 和思考模式支持。

## Latest: v0.2.2

### Fixes
- **#34**: Fixed thinking mode markup tags — `<think/>`/`</think/>` → `<think>`/`</think>`, aligned 1:1 with [dify-official-plugins](https://github.com/langgenius/dify-official-plugins) openai_api_compatible
- **#31**: Fixed `extra_body` parameter delivery — JSON contents now merged into top-level request body instead of nested under `"extra_body"` key

### Features (since v0.2.0)
- **extra_body**: Pass any vLLM extra parameters directly as JSON
- **Thinking Mode**: `enable_thinking` toggle with `compatibility_mode` (strict/extended)
- **Reasoning Effort**: Support `reasoning_effort` (none/low/medium/high), natively supported by vLLM
- **Compatibility Mode**: Extended mode injects `chat_template_kwargs`, `thinking`, `enable_thinking` at top level
- **Structured Output**: Support `response_format`, `json_schema`, `reasoning_format`
- **Thinking Content Filter**: Auto-filter `<think>...</think>` when thinking is disabled
- **Thinking Content Cleanup**: Strip thinking content from history before requests
- **vLLM Reasoning Field**: Priority read `reasoning` (vLLM >= 0.17.1), fallback to `reasoning_content`

### v0.2.0 Baseline
> **Breaking Change**: Removed all legacy parameters, use `extra_body` for all extra parameter needs.

## Usage

### Add model
Same as OpenAI-API-compatible, select "Vllm" provider:

![add model](./_assets/add_model.png)

### Configure extra_body
Pass extra parameters via the `extra_body` JSON text field:

![use guided](./_assets/use_guided.png)

Example:
```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

## Repo
https://github.com/yangyaofei/dify-vllm-provider

# vllm-openai dify provider plugin to support extra parameters


## **NOTE!!!**
**This plugin is a extension for official OpenAI-API-compatible,** 
**provide features for [extra parameters in vLLM's OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters).**

**本插件是在官方 OpenAI-API-compatible 基础上构建, 用于提供[vLLM's OpenAI-Compatible Server 中的 extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters)**.
**若没有使用上述 extra parameters 中的相关的特性, 请使用官方 OpenAI-API-compatible 插件.**

## **version 0.2.0 notice**
由于 Dify 的 openai compatible provider 和 vLLM's openai compatible server 都已完整支持结构化输出, 因此本项目的原有的主要功能支持已经没有必要.
在 0.2.0 中会去除对原有任何参数的支持, 转而直接使用 `extra_body` 来传递 extra parameters. 

## **version 0.2.0 notice**
Due to the change of Dify's openai compatible provider and vLLM's openai compatible server support the CFG(Classifier-Free Guidance) features.
So, it's not necessary to do structured output here, and so as other features.
In 0.2.0, the plugin will remove all parameters and add `extra_body` to pass extra parameters directly to vLLM backend.

## **version 0.2.1 notice**
基于官方 OpenAI-API-compatible 插件的最新实现，增加了以下功能：
- **Thinking Mode**: 支持 `enable_thinking` 开关，适配 vLLM 的 `chat_template_kwargs` 和 `reasoning` 响应字段
- **Compatibility Mode**: 新增 `compatibility_mode` 配置（strict/extended），控制是否注入额外参数
- **Reasoning Effort**: 支持 `reasoning_effort` 参数（none/low/medium/high），vLLM 原生支持
- **Structured Output**: 支持 `response_format`、`json_schema`、`reasoning_format` 参数
- **Thinking Content Filter**: 当 thinking 关闭时自动过滤 `<think/>` 内容
- 保留 `extra_body` 作为 vLLM extra parameters 的通用传递方式

## **version 0.2.1 notice**
Based on the latest official OpenAI-API-compatible plugin implementation, the following features have been added:
- **Thinking Mode**: Support `enable_thinking` toggle, compatible with vLLM's `chat_template_kwargs` and `reasoning` response field
- **Compatibility Mode**: New `compatibility_mode` config (strict/extended), controls whether to inject extra parameters
- **Reasoning Effort**: Support `reasoning_effort` parameter (none/low/medium/high), natively supported by vLLM
- **Structured Output**: Support `response_format`, `json_schema`, `reasoning_format` parameters
- **Thinking Content Filter**: Automatically filter `<think/>` content when thinking is disabled
- Keep `extra_body` as the generic way to pass vLLM extra parameters

## Repo
https://github.com/yangyaofei/dify-vllm-provider

## Description

The vllm [openAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) has [extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters).**
, The openai compatible provider in Dify can not do this.

This plugin provide a vllm-openai provider upon Dify's openai compatible provider with `extra_body`
### Add model same as openai compatible with vLLM-openai backend

![`add model`](./_assets/add_model.png)

### Config model guided with `extra_body`

![`use guided`](./_assets/use_guided.png)
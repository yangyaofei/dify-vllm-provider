# vllm-openai dify provider plugin to support guided generate

## **NOTE!!!**
**This plugin is a extension for official OpenAI-API-compatible,** 
**provide features for [extra parameters in vLLM's OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters).**
**Only extra parameters featured model will be implemented here, Please use official OpenAI-API-compatible you don't have such needs.**

**本插件是在官方 OpenAI-API-compatible 基础上构建, 用于提供[vLLM's OpenAI-Compatible Server 中的 extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters)**
**的调用(现主要集中在 CFG 即 Guided Generate 上). 若没有使用上述 extra parameters 中的相关的特性, 请使用官方 OpenAI-API-compatible 插件.**

## Repo
https://github.com/yangyaofei/dify-vllm-provider

## Description

The vllm [openAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 
can make CFG(Classifier-Free Guidance) with [outlines](https://github.com/dottxt-ai/outlines) or 
[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) with `Extra Parameters` like `guided_json`, `guided_regex`.
The openai compatible provider in Dify can not do this, but like gpt-4o can do same thing wit_assets/add_model.pngh json flag.

This plugin provide a vllm-openai provider upon Dify's openai compatible provider with `guided_json`, `guided_regex`, `guided_grammar`.

### Add model same as openai compatible with vLLM-openai backend

![`add model`](./_assets/add_model.png)

### Config model guided with `json schema`, `regex`, `grammar`

![`use guided`](./_assets/use_guided.png)

### Config model with Dify's structured output

When config with this, plugin will directly use `guided_json` to the vLLM backend.

![`use structured output`](./_assets/structured_output.png)

### Config model thinking mode

You can disable/enable thinking mode for supported models (Qwen3, DeepSeek-R1).

![`config thinking mode`](./_assets/deep_think.png)

## Dynamic Request guided

The guided param may various in same workflow, but Dify doesn't give a extra param in LLM to pass it.

This comes little hacky, when the prompt has 2nd part and it's assistant, this provider will use the assistant part as guided param.
If it can be parsed as json.

The structure of param can be found in `GuidedParam` class.

You can enable it with config:

![`enable dynamic request guided`](./_assets/dynamic_request_guided.png)

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
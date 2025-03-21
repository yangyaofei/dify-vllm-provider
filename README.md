# vllm-openai dify provider plugin to support guided generate

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


## Dynamic Request guided

The guided param may various in same workflow, but Dify doesn't give a extra param in LLM to pass it.

This comes little hacky, when the prompt has 2nd part and it's assistant, this provider will use the assistant part as guided param.
If it can be parsed as json.

The structure of param can be found in `GuidedParam` class.

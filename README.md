# A vllm-openai provider for Dify 

The vllm [openAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 
can make CFG(Classifier-Free Guidance) with [outlines](https://github.com/dottxt-ai/outlines) or 
[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) with `Extra Parameters` like `guided_json`, `guided_regex`.
The openai compatible provider in Dify can not do this, but like gpt-4o can do same thing with json flag.

This provider provide a vllm provider upon Dify's openai compatible provider with `guided_json`, `guided_regex`, `guided_grammar`.

## Dynamic Request guided

The guided param may various in same workflow, but Dify doesn't give a extra param in LLM to pass it.

This comes little hacky, when the prompt has 2nd part and it's assistant, this provider will use the assistant part as guided param.
If it can be parsed as json.

The structure of param can be found in `GuidedParam` class.

## Use in Docker

add volume in `docker-compose.yaml`:

```yaml

api:
  image: langgenius/dify-api:XXX
  # add config in volumes
  volumes:
    # add this
    - path_to_project/vllm:/app/api/core/model_runtime/model_providers/vllm in 
```

restart docker-compose , you will see it
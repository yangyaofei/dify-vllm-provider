import enum
import json
import logging
from collections.abc import Generator, Sequence
from typing import Optional, Union, Dict

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    ParameterRule, ParameterType,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool, PromptMessageRole,
)
from pydantic import BaseModel, ValidationError

# add logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


class GuidedType(str, enum.Enum):
    JSON = "guided_json"
    REGEX = "guided_regex"
    CHOICE = "guided_choice"
    GRAMMAR = "guided_grammar"


class GuidedParam(BaseModel):
    param_type: GuidedType
    param: Union[Dict, str]


class VllmLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    Model class for vllm large language model.
    """

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[Sequence[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        logger.info(f"Model parameters: {model_parameters}")
        logger.info(f"Prompt messages: {prompt_messages}")
        # check dynamic_request_guided
        if (model_parameters.pop("dynamic_request_guided", False)
                and len(prompt_messages) >= 2
                and prompt_messages[1].role == PromptMessageRole.ASSISTANT):
            if type(prompt_messages[1].content) is str:
                try:
                    param = GuidedParam(**json.loads(prompt_messages[1].content))
                    model_parameters.update({param.param_type: json.dumps(param.param, ensure_ascii=False)})
                    prompt_messages.pop(1)
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Error in extract param from prompt, bypass, error msg: {e}")
                except Exception as e:
                    logger.warning(f"Error in extract param from prompt, bypass, error msg: {e}")

        # In the vllm framework, when using deep-thinking models, the thinking feature is enabled by default.
        # Therefore, when no parameters are passed, vllm's responses will include thought results.
        if "enable_thinking" in model_parameters:
            model_parameters["chat_template_kwargs"] = {
                "enable_thinking": model_parameters.get("enable_thinking", False)}

        if "json_schema" in model_parameters:
            model_parameters["guided_json"] = model_parameters.pop("json_schema")

        logger.info(f"Request Model parameters: {model_parameters}")
        return super()._invoke(
            model, credentials, prompt_messages, model_parameters,
            tools, stop, stream, user
        )

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)
        entity.parameter_rules += [
            ParameterRule(
                name="json_schema",
                use_template="json_schema",
                type=ParameterType.STRING,
                label=I18nObject(en_US="json_schema")
            ),
            ParameterRule(
                name="enable_thinking",
                label=I18nObject(en_US="Deep thinking", zh_Hans="深度思考"),
                help=I18nObject(
                    en_US="Whether to enable deep thinking, applicable to various thinking mode models deployed on reasoning frameworks such as vLLM for example Qwen3 and deepseek-r1.",
                    zh_Hans="是否开启深度思考，适用于vLLM等推理框架部署的多种思考模式模型，例如Qwen3和deepseek-r1。",
                ),
                type=ParameterType.BOOLEAN,
                default=False,
                required=False
            ),
            ParameterRule(
                name="guided_json",
                label=I18nObject(en_US="guided_json"),
                help=I18nObject(en_US="guided_json in vllm, If specified, the output will follow the JSON schema."),
                type=ParameterType.TEXT,
                required=False
            ),
            ParameterRule(
                name="guided_regex",
                label=I18nObject(en_US="guided_regex"),
                help=I18nObject(en_US="If specified, the output will follow the regex pattern."),
                type=ParameterType.TEXT,
                required=False
            ),
            ParameterRule(
                name="guided_grammar",
                label=I18nObject(en_US="guided_grammar"),
                help=I18nObject(en_US="If specified, the output will follow the context free grammar."),
                type=ParameterType.TEXT,
                required=False
            ),
            ParameterRule(
                name="dynamic_request_guided",
                label=I18nObject(en_US="Dynamic Request Guided"),
                help=I18nObject(
                    en_US="If set to true, when the prompt has 2nd part and it's assistant, this provider will use the assistant part as guided param."
                ),
                type=ParameterType.BOOLEAN,
                default=False,
                required=False
            )

        ]
        return entity

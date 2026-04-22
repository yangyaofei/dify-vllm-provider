import json
import re
import logging
from contextlib import suppress
from typing import Optional, Union, Generator, List

from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    I18nObject,
    ModelFeature,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    AssistantPromptMessage,
)
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


def _validate_extra_body(model_parameters: dict):
    extra_body = model_parameters.get("extra_body")
    if not extra_body:
        return
    if isinstance(extra_body, dict):
        return
    if not isinstance(extra_body, str):
        logger.error(f"extra_body must be valid json str, got {type(extra_body).__name__}, bypass")
        model_parameters.pop("extra_body", None)
        return
    try:
        model_parameters["extra_body"] = json.loads(extra_body)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid extra_body JSON, bypass: {e}")
        model_parameters.pop("extra_body", None)


class VllmLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    vLLM large language model with extra_body support.
    Based on official OpenAI-API-compatible plugin with vLLM-specific enhancements.
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
    """

    _THINK_PATTERN = re.compile(r"^<think\b[^>]*>.*?</think\s*>\s*", re.DOTALL)

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """
        Override base wrapper to support vLLM's 'reasoning' field (vLLM >= 0.17.1)
        and legacy 'reasoning_content' field, emitting <think/> blocks
        compatible with Dify's downstream filters.
        """
        reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
        content_piece = delta.get("content") or ""

        if reasoning_piece:
            if not is_reasoning:
                output = f"<think/>\n{reasoning_piece}"
                is_reasoning = True
            else:
                output = str(reasoning_piece)
        elif is_reasoning:
            is_reasoning = False
            output = f"\n</think/>{content_piece}"
        else:
            output = content_piece

        return output, is_reasoning

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        structured_output_support = credentials.get("structured_output_support", "not_supported")
        if structured_output_support == "supported":
            entity.parameter_rules.append(
                ParameterRule(
                    name=DefaultParameterName.RESPONSE_FORMAT.value,
                    label=I18nObject(en_US="Response Format", zh_Hans="回复格式"),
                    help=I18nObject(
                        en_US="Specifying the format that the model must output.",
                        zh_Hans="指定模型必须输出的回复格式。",
                    ),
                    type=ParameterType.STRING,
                    options=["text", "json_object", "json_schema"],
                    required=False,
                )
            )
            entity.parameter_rules.append(
                ParameterRule(
                    name="reasoning_format",
                    label=I18nObject(en_US="Reasoning Format", zh_Hans="推理格式"),
                    help=I18nObject(
                        en_US="Specifying the format that the model must output reasoning.",
                        zh_Hans="指定模型必须输出的推理格式。",
                    ),
                    type=ParameterType.STRING,
                    options=["none", "auto", "deepseek", "deepseek-legacy"],
                    required=False,
                )
            )
            entity.parameter_rules.append(
                ParameterRule(
                    name=DefaultParameterName.JSON_SCHEMA.value,
                    use_template=DefaultParameterName.JSON_SCHEMA.value,
                )
            )

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(
                en_US=credentials["display_name"], zh_Hans=credentials["display_name"]
            )

        agent_thought_support = credentials.get("agent_thought_support", "not_supported")

        if agent_thought_support in ["supported", "only_thinking_supported"]:
            if ModelFeature.AGENT_THOUGHT not in entity.features:
                entity.features.append(ModelFeature.AGENT_THOUGHT)

        if agent_thought_support == "supported":
            entity.parameter_rules.append(
                ParameterRule(
                    name="enable_thinking",
                    label=I18nObject(en_US="Thinking mode", zh_Hans="思考模式"),
                    help=I18nObject(
                        en_US="Whether to enable thinking mode, applicable to various thinking mode models deployed on vLLM and SGLang.",
                        zh_Hans="是否开启思考模式，适用于vLLM和SGLang等推理框架部署的多种思考模式模型。",
                    ),
                    type=ParameterType.BOOLEAN,
                    required=False,
                )
            )

        if agent_thought_support in ["supported", "only_thinking_supported"]:
            entity.parameter_rules.append(
                ParameterRule(
                    name="reasoning_effort",
                    label=I18nObject(en_US="Reasoning effort", zh_Hans="推理力度"),
                    help=I18nObject(
                        en_US="Constrains effort on reasoning for reasoning models. vLLM supports none/low/medium/high.",
                        zh_Hans="限制推理模型的推理力度。vLLM 支持 none/low/medium/high。",
                    ),
                    type=ParameterType.STRING,
                    options=["none", "low", "medium", "high"],
                    required=False,
                )
            )

        entity.parameter_rules.append(
            ParameterRule(
                name="extra_body",
                type=ParameterType.TEXT,
                label=I18nObject(en_US="Extra Body", zh_Hans="额外参数"),
                help=I18nObject(
                    en_US="JSON object passed as extra_body to vLLM OpenAI-compatible API. "
                          "See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters",
                    zh_Hans="作为 extra_body 传递给 vLLM OpenAI 兼容 API 的 JSON 对象。"
                            "参考 https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters",
                ),
            )
        )

        return entity

    @classmethod
    def _drop_analyze_channel(cls, prompt_messages: List[PromptMessage]) -> None:
        """Remove thinking content from assistant messages for better performance."""
        for p in prompt_messages:
            if not isinstance(p, AssistantPromptMessage):
                continue
            if not isinstance(p.content, str):
                continue
            if not p.content.startswith("<think"):
                continue
            new_content = cls._THINK_PATTERN.sub("", p.content, count=1)
            if new_content != p.content:
                p.content = new_content

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        if model_parameters.get("response_format") == "json_schema":
            json_schema_str = model_parameters.get("json_schema")
            if json_schema_str:
                structured_output_prompt = (
                    "Your response must be a JSON object that validates against the following JSON schema, and nothing else.\n"
                    f"JSON Schema: ```json\n{json_schema_str}\n```"
                )
                existing_system_prompt = next(
                    (p for p in prompt_messages if p.role == PromptMessageRole.SYSTEM), None
                )
                if existing_system_prompt:
                    existing_system_prompt.content = (
                        structured_output_prompt + "\n\n" + existing_system_prompt.content
                    )
                else:
                    prompt_messages.insert(0, SystemPromptMessage(content=structured_output_prompt))

        agent_thought_support = credentials.get("agent_thought_support", "not_supported")
        enable_thinking_value = None
        if agent_thought_support == "only_thinking_supported":
            enable_thinking_value = True
        elif agent_thought_support == "not_supported":
            enable_thinking_value = False
        else:
            user_enable_thinking = model_parameters.pop("enable_thinking", None)
            if user_enable_thinking is not None:
                enable_thinking_value = bool(user_enable_thinking)

        compatibility_mode = credentials.get("compatibility_mode", "strict")
        extended_mode = compatibility_mode == "extended"

        if enable_thinking_value is not None and extended_mode:
            chat_template_kwargs = model_parameters.setdefault("chat_template_kwargs", {})
            chat_template_kwargs["enable_thinking"] = enable_thinking_value
            chat_template_kwargs["thinking"] = enable_thinking_value

            model_parameters["thinking"] = {
                "type": "enabled" if enable_thinking_value else "disabled"
            }
            model_parameters["enable_thinking"] = enable_thinking_value

        reasoning_effort_value = model_parameters.pop("reasoning_effort", None)
        if enable_thinking_value is True and reasoning_effort_value is not None:
            model_parameters["reasoning_effort"] = reasoning_effort_value
            if extended_mode:
                chat_template_kwargs = model_parameters.setdefault("chat_template_kwargs", {})
                chat_template_kwargs["reasoning_effort"] = reasoning_effort_value

        with suppress(Exception):
            self._drop_analyze_channel(prompt_messages)

        _validate_extra_body(model_parameters)

        result = super()._invoke(
            model, credentials, prompt_messages, model_parameters,
            tools, stop, stream, user
        )

        if enable_thinking_value is False:
            if stream:
                return self._filter_thinking_stream(result)
            else:
                return self._filter_thinking_result(result)

        return result

    def _filter_thinking_result(self, result: LLMResult) -> LLMResult:
        """Filter thinking content from non-streaming result."""
        if result.message and result.message.content:
            content = result.message.content
            if isinstance(content, str) and content.startswith("<think"):
                filtered_content = self._THINK_PATTERN.sub("", content, count=1)
                if filtered_content != content:
                    result.message.content = filtered_content
        return result

    def _filter_thinking_stream(self, stream: Generator) -> Generator:
        """Filter thinking content from streaming result."""
        buffer = ""
        in_thinking = False
        thinking_started = False

        for chunk in stream:
            if chunk.delta and chunk.delta.message and chunk.delta.message.content:
                content = chunk.delta.message.content
                buffer += content

                if not thinking_started and buffer.startswith("<think"):
                    in_thinking = True
                    thinking_started = True

                if in_thinking and "</think" in buffer:
                    end_idx = buffer.find("</think")
                    end_idx = buffer.find(">", end_idx) + 1
                    while end_idx < len(buffer) and buffer[end_idx].isspace():
                        end_idx += 1
                    buffer = buffer[end_idx:]
                    in_thinking = False
                    thinking_started = False
                    if buffer:
                        chunk.delta.message.content = buffer
                        buffer = ""
                        yield chunk
                    continue

                if not in_thinking:
                    yield chunk
                    buffer = ""
            else:
                yield chunk

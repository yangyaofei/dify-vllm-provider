import enum
import json
import logging
from collections.abc import Generator, Sequence
from typing import Optional, Union, Dict

from core.model_runtime.callbacks.base_callback import Callback
from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.llm_entities import LLMResult
from core.model_runtime.entities.message_entities import (
    PromptMessage,
    PromptMessageTool, PromptMessageRole,
)
from core.model_runtime.entities.model_entities import (
    AIModelEntity,
)
from core.model_runtime.entities.model_entities import (
    ParameterRule,
    ParameterType
)
from core.model_runtime.model_providers.openai_api_compatible.llm.llm import OAIAPICompatLargeLanguageModel
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class GuidedType(str, enum.Enum):
    JSON = "guided_json"
    REGEX = "guided_regex"
    CHOICE = "guided_choice"
    GRAMMAR = "guided_grammar"


class GuidedParam(BaseModel):
    param_type: GuidedType
    param: Union[Dict, str]


class OAIAPICompatVllmLargeLanguageModel(OAIAPICompatLargeLanguageModel):
    """
    Model class for OpenAI Vllm
    """

    def invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: Optional[dict] = None,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[Sequence[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
            callbacks: Optional[list[Callback]] = None,
    ) -> Union[LLMResult, Generator]:
        if len(prompt_messages) >= 2 and prompt_messages[1].role == PromptMessageRole.ASSISTANT:
            try:
                param = GuidedParam(**json.loads(prompt_messages[1].content))
                model_parameters.update({param.param_type: json.dumps(param.param, ensure_ascii=False)})
                prompt_messages.pop(1)
            except json.JSONDecodeError | ValidationError:
                # do nothing if it's not a valid config
                pass

        return super().invoke(
            model, credentials, prompt_messages, model_parameters,
            tools, stop, stream, user, callbacks
        )

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)
        entity.parameter_rules += [
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
            )

        ]
        return entity

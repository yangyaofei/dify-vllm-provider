import json
import logging
from collections.abc import Generator
from decimal import Decimal
from typing import Optional, Union
from urllib.parse import urljoin

import requests

from core.model_runtime import entities
from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.llm_entities import LLMMode, LLMResult, LLMResultChunk, LLMResultChunkDelta
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageFunction,
    PromptMessageTool,
)
from core.model_runtime.entities.model_entities import (
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
    PriceConfig,
)
from core.model_runtime.errors.invoke import InvokeError
from core.model_runtime.model_providers.openai_api_compatible.llm.llm import OAIAPICompatLargeLanguageModel
from core.model_runtime.utils import helper

logger = logging.getLogger(__name__)


class OAIAPICompatVllmLargeLanguageModel(OAIAPICompatLargeLanguageModel):
    """
    Model class for OpenAI Vllm
    """

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
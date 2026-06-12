import json
from dataclasses import replace
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai import ModelSettings, ModelProfile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings as _ModelSettings
import json_repair

import logger
from app_config import get_env, http_chat_completions_thinking_extras


class JsonRepairOpenAIChatModel(OpenAIChatModel):
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: _ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        return self._repair_tool_calls_json(response)

    def _ensure_valid_json(self, args_str: str, tool_name: str) -> str:
        """保证返回的字符串一定是合法 JSON。先校验，再 json_repair，最后 fallback 标记。"""
        try:
            json.loads(args_str)
            return args_str
        except json.JSONDecodeError:
            pass

        logger.warning("tool=%s args 非法JSON: %s", tool_name, args_str)

        try:
            fixed = json_repair.loads(args_str)
            return json.dumps(fixed, ensure_ascii=False)
        except Exception:
            pass

        logger.error("tool=%s args 无法修复，fallback 为 _args_corrupted 标记", tool_name)
        return json.dumps({"_args_corrupted": True, "tool": tool_name}, ensure_ascii=False)

    def _repair_tool_calls_json(self, response: ModelResponse) -> ModelResponse:
        """修复响应中所有工具调用的 JSON 参数"""
        repaired_parts = []

        for part in response.parts:
            if isinstance(part, ToolCallPart):
                try:
                    original_args = part.args
                    if isinstance(original_args, dict):
                        repaired_args = json.dumps(original_args, ensure_ascii=False)
                    else:
                        repaired_args = self._ensure_valid_json(original_args, part.tool_name)
                    part = replace(part, args=repaired_args)
                except Exception as e:
                    logger.warning("repair tool call JSON failed for %s: %s", part.tool_name, e)
                    # fallback：用 _ensure_valid_json 保证不存入坏数据
                    if isinstance(part.args, str):
                        safe_args = self._ensure_valid_json(part.args, part.tool_name)
                        part = replace(part, args=safe_args)
            repaired_parts.append(part)

        return replace(response, parts=repaired_parts)

    def _process_thinking(self, message: Any) -> list[ThinkingPart] | None:
        """DeepSeek thinking + tools: reasoning_content must round-trip even when empty, or API returns 400."""
        inherited = super()._process_thinking(message)
        if inherited:
            return inherited
        if not getattr(message, 'tool_calls', None):
            return None
        profile = OpenAIModelProfile.from_profile(self.profile)
        if (
            profile.openai_chat_thinking_field
            and profile.openai_chat_send_back_thinking_parts == 'field'
        ):
            field = profile.openai_chat_thinking_field
            return [
                ThinkingPart(id=field, content='', provider_name=self.system),
            ]
        return None


def _openai_compatible_thinking_profile(model_name: str) -> ModelProfile:
    provider_profile = OpenAIProvider.model_profile(model_name)
    profile = OpenAIModelProfile.from_profile(provider_profile).update(
        deepseek_model_profile(model_name)
    )
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_object_output=True,
        supports_thinking=True,
        openai_chat_thinking_field='reasoning_content',
        openai_chat_send_back_thinking_parts='field',
    ).update(profile)


def create_model(model_name: str, parameter: dict):
    api_key = get_env("API_KEY", warn=False) or None
    api_base = get_env("BASE_URL", warn=False) or None
    param = dict(parameter)

    if 'gemini' in model_name:
        del param["thinking"]
        provider = GoogleProvider(base_url='https://api.zhizengzeng.com/google', api_key=api_key)
        return GoogleModel(model_name, provider=provider, settings=ModelSettings(**param))

    if 'claude' in model_name:
        del param["thinking"]
        provider = AnthropicProvider(base_url='https://api.zhizengzeng.com/anthropic', api_key=api_key)
        return AnthropicModel(model_name, provider=provider, settings=ModelSettings(**param))

    if any(m in model_name.lower() for m in ('deepseek', 'kimi')):
        eb = param.get('extra_body')
        extra = dict(eb) if isinstance(eb, dict) else {}
        extra.update(http_chat_completions_thinking_extras(param))
        del param["thinking"]
        if extra:
            param['extra_body'] = extra
        provider = OpenAIProvider(base_url=api_base, api_key=api_key)
        profile = _openai_compatible_thinking_profile(model_name)
    else:
        del param["thinking"]
        provider = OpenAIProvider(base_url=api_base, api_key=api_key)
        profile = None
    return JsonRepairOpenAIChatModel(
        model_name,
        provider=provider,
        profile=profile,
        settings=ModelSettings(**param),
    )

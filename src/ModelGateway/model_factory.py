import json
import httpx
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
from pydantic_ai.models import ModelRequestParameters, create_async_http_client
from pydantic_ai.settings import ModelSettings as _ModelSettings
import json_repair

from infra import logger
from config.app_config import get_env, http_chat_completions_thinking_extras, unified_thinking_setting
from infra.shared_http import get_client

_HTTP_KEY = "model"


class JsonRepairOpenAIChatModel(OpenAIChatModel):
    class _ReasoningContentMapContext(OpenAIChatModel._MapModelResponseContext):
        """Replay 兜底：带 tool_calls 的 assistant 消息缺 reasoning_content 时补空字符串。"""

        def _into_message_param(self):
            message_param = super()._into_message_param()
            if message_param is None:
                return None
            field = self._model._reasoning_content_field_name()
            if (
                field
                and field not in message_param
                and message_param.get("tool_calls")
            ):
                message_param[field] = ""
            return message_param

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: _ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        return self._repair_tool_calls_json(response)

    def _map_model_response(self, message: ModelResponse):
        return self._ReasoningContentMapContext(self).map_assistant_message(message)

    def _reasoning_content_field_name(self) -> str | None:
        profile = OpenAIModelProfile.from_profile(self.profile)
        if profile.openai_chat_send_back_thinking_parts != "field":
            return None
        return profile.openai_chat_thinking_field or None

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


def _get_shared_http_client() -> httpx.AsyncClient:
    timeout = int(float(get_env("MODEL_HTTP_TIMEOUT", warn=False) or 300))
    return get_client(
        _HTTP_KEY,
        lambda: create_async_http_client(timeout=timeout, connect=10),
    )


def create_model(model_name: str, parameter: dict):
    api_key = get_env("API_KEY", warn=False) or None
    api_base = get_env("BASE_URL", warn=False) or None
    http_client = _get_shared_http_client()
    param = dict(parameter)
    thinking_extras = http_chat_completions_thinking_extras(param)
    thinking_level = unified_thinking_setting(param)
    param.pop("thinking", None)
    param.pop("reasoning_effort", None)

    if 'gemini' in model_name:
        param["thinking"] = thinking_level
        provider = GoogleProvider(base_url='https://api.zhizengzeng.com/google', api_key=api_key, http_client=http_client)
        return GoogleModel(model_name, provider=provider, settings=ModelSettings(**param))

    if 'claude' in model_name:
        param["thinking"] = thinking_level
        provider = AnthropicProvider(base_url='https://api.zhizengzeng.com/anthropic', api_key=api_key, http_client=http_client)
        return AnthropicModel(model_name, provider=provider, settings=ModelSettings(**param))

    if any(m in model_name.lower() for m in ('deepseek', 'kimi')):
        eb = param.get('extra_body')
        extra = dict(eb) if isinstance(eb, dict) else {}
        extra.update(thinking_extras)
        if extra:
            param['extra_body'] = extra
        profile = _openai_compatible_thinking_profile(model_name)
    else:
        profile = None
    provider = OpenAIProvider(base_url=api_base, api_key=api_key, http_client=http_client)
    return JsonRepairOpenAIChatModel(
        model_name,
        provider=provider,
        profile=profile,
        settings=ModelSettings(**param),
    )

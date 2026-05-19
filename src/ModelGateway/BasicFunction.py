import json
from openai.types import chat as oa_chat
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai import Agent, ModelSettings, ModelProfile
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
        self._log_missing_reasoning_content(messages)
        response = await super().request(messages, model_settings, model_request_parameters)
        return self._repair_tool_calls_json(response)

    def _log_missing_reasoning_content(self, messages: list[ModelMessage]) -> None:
        profile = OpenAIModelProfile.from_profile(self.profile)
        field = profile.openai_chat_thinking_field
        if not field or profile.openai_chat_send_back_thinking_parts != 'field':
            return

        missing_indices: list[int] = []
        for idx, message in enumerate(messages):
            if not isinstance(message, ModelResponse):
                continue
            mapped = self._map_model_response(message)
            if mapped.get('role') != 'assistant':
                continue
            value = mapped.get(field)
            if not isinstance(value, str):
                missing_indices.append(idx)

        if missing_indices:
            logger.debug(
                "thinking history missing `%s` on assistant messages: count=%s indices=%s",
                field,
                len(missing_indices),
                missing_indices,
            )

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
                    part = ToolCallPart(
                        tool_name=part.tool_name,
                        args=repaired_args,
                        tool_call_id=part.tool_call_id,
                        id=part.id,
                        provider_details=part.provider_details,
                    )
                except Exception as e:
                    logger.warning("repair tool call JSON failed for %s: %s", part.tool_name, e)
                    # fallback：用 _ensure_valid_json 保证不存入坏数据
                    if isinstance(part.args, str):
                        safe_args = self._ensure_valid_json(part.args, part.tool_name)
                        part = ToolCallPart(
                            tool_name=part.tool_name,
                            args=safe_args,
                            tool_call_id=part.tool_call_id,
                            id=part.id,
                            provider_details=part.provider_details,
                        )
            repaired_parts.append(part)

        return ModelResponse(
            parts=repaired_parts,
            usage=response.usage,
            model_name=response.model_name,
            timestamp=response.timestamp,
            provider_name=response.provider_name,
            provider_details=response.provider_details,
            provider_response_id=response.provider_response_id,
            finish_reason=response.finish_reason,
            run_id=response.run_id,
            metadata=response.metadata,
        )

    def _process_thinking(self, message: oa_chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        """DeepSeek thinking + tools: reasoning_content must round-trip even when empty, or API returns 400."""
        inherited = super()._process_thinking(message)
        if inherited:
            return inherited
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


class ThinkingProvider(OpenAIProvider):
    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = deepseek_model_profile(model_name)
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
        provider = GoogleProvider(
            base_url='https://api.zhizengzeng.com/google',
            api_key=api_key,
        )
        return GoogleModel(model_name, provider=provider, settings=ModelSettings(**param))
    elif 'claude' in model_name:
        del param["thinking"]
        provider = AnthropicProvider(
            base_url='https://api.zhizengzeng.com/anthropic',
            api_key=api_key,
        )
        return AnthropicModel(model_name, provider=provider, settings=ModelSettings(**param))
    else:
        thinking_models = ['deepseek', 'kimi']
        use_thinking_provider = any(m in model_name.lower() for m in thinking_models)
        
        if use_thinking_provider:
            provider = ThinkingProvider(base_url=api_base, api_key=api_key)
            extra: dict = {}
            eb = param.get('extra_body')
            if isinstance(eb, dict):
                extra = dict(eb)
            extra.update(http_chat_completions_thinking_extras(param))
            del param["thinking"]
            if extra:
                param['extra_body'] = extra
            settings_kw = param
        else:
            del param["thinking"]
            provider = OpenAIProvider(
                base_url=api_base,
                api_key=api_key,
            )
            settings_kw = param
        return JsonRepairOpenAIChatModel(
            model_name,
            provider=provider,
            settings=ModelSettings(**settings_kw)
        )


def create_agent(model_name: str, parameter: dict, tools: list, system_prompt: str):
    if parameter is None:
        parameter = {
            "temperature": 1.0,
            "max_tokens": 32768,
            "reasoning_effort": False,
            "thinking": "disabled",
        }

    model = create_model(model_name, parameter)
    wrapped_tools = logger.wrap_tools_for_user_notify(list(tools)) if tools else tools
    agent = Agent(
        model,
        tools=wrapped_tools,
        system_prompt=system_prompt,
    )
    return agent
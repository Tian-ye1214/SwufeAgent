from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai import Agent, ModelSettings, ModelProfile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.messages import ModelResponse, ToolCallPart
import json_repair
import os
load_dotenv()


class JsonRepairOpenAIChatModel(OpenAIChatModel):
    async def request(self, *args, **kwargs) -> ModelResponse:
        response = await super().request(*args, **kwargs)
        return self._repair_tool_calls_json(response)
    
    def _truncate_long_content(self, json_str: str, max_content_length: int = 8000) -> str:
        """对于 write_file 等工具，截断过长的 content 字段"""
        try:
            data = json_repair.loads(json_str)
            if isinstance(data, dict) and 'content' in data:
                content = data['content']
                if isinstance(content, str) and len(content) > max_content_length:
                    data['content'] = content[:max_content_length] + "\n\n... [内容被截断，原长度: " + str(len(content)) + " 字符] ..."
            return json_repair.dumps(data)
        except Exception:
            return json_str
    
    def _repair_truncated_json(self, json_str: str, tool_name: str) -> str:
        """
        尝试修复被截断的 JSON 字符串。
        当模型输出被截断时，JSON 可能是不完整的。
        """
        if not json_str or not isinstance(json_str, str):
            return json_str

        try:
            repaired = json_repair.loads(json_str)
            return json_repair.dumps(repaired)
        except Exception:
            pass

        json_str = json_str.strip()
        if tool_name == 'write_file':
            try:
                import re
                name_match = re.search(r'"name"\s*:\s*"([^"]*)"', json_str)
                if name_match:
                    file_name = name_match.group(1)
                    return json_repair.dumps({
                        "name": file_name,
                        "content": "[ERROR: Content was truncated due to length. Please write the file in smaller chunks or use a shorter content. Maximum recommended content length is 8000 characters.]"
                    })
            except Exception:
                pass

        try:
            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')
            in_string = False
            escape_next = False
            for char in json_str:
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
            if in_string:
                json_str += '"'

            json_str += ']' * open_brackets
            json_str += '}' * open_braces

            repaired = json_repair.loads(json_str)
            return json_repair.dumps(repaired)
        except Exception:
            pass

        return json_str
    
    def _repair_tool_calls_json(self, response: ModelResponse) -> ModelResponse:
        """修复响应中所有工具调用的 JSON 参数"""
        repaired_parts = []
        
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                try:
                    original_args = part.args
                    if isinstance(original_args, str):
                        repaired_args = self._repair_truncated_json(original_args, part.tool_name)

                        if part.tool_name == 'write_file':
                            repaired_args = self._truncate_long_content(repaired_args)
                            
                    elif isinstance(original_args, dict):
                        repaired_args = json_repair.dumps(original_args)
                    else:
                        repaired_args = original_args
                    part = ToolCallPart(
                        tool_name=part.tool_name,
                        args=repaired_args,
                        tool_call_id=part.tool_call_id,
                        id=part.id,
                        provider_details=part.provider_details,
                    )
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to repair tool call JSON for {part.tool_name}: {e}")
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


class ThinkingProvider(OpenAIProvider):
    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = deepseek_model_profile(model_name)
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            supports_json_object_output=True,
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        ).update(profile)


def create_model(model_name: str, parameter: dict):
    if 'gemini' in model_name:
        provider = GoogleProvider(
            base_url='https://api.zhizengzeng.com/google',
            api_key=os.environ.get('API_KEY')
        )
        return GoogleModel(model_name, provider=provider, settings=ModelSettings(**parameter))
    elif 'claude' in model_name:
        provider = AnthropicProvider(
            base_url='https://api.zhizengzeng.com/anthropic',
            api_key=os.environ.get('API_KEY')
        )
        return AnthropicModel(model_name, provider=provider, settings=ModelSettings(**parameter))
    else:
        thinking_models = ['deepseek', 'kimi']
        use_thinking_provider = any(m in model_name.lower() for m in thinking_models)
        
        if use_thinking_provider:
            provider = ThinkingProvider(base_url=os.environ.get('BASE_URL'), api_key=os.environ.get('API_KEY'))
        else:
            provider = OpenAIProvider(
                base_url=os.environ.get('BASE_URL'),
                api_key=os.environ.get('API_KEY')
            )
        return JsonRepairOpenAIChatModel(
            model_name,
            provider=provider,
            settings=ModelSettings(**parameter)
        )


def create_agent(model_name: str, parameter: dict, tools: list, system_prompt: str):
    if parameter is None:
        parameter = {
            "temperature": 1.0,
            "max_tokens": 32768,
        }

    model = create_model(model_name, parameter)
    agent = Agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent
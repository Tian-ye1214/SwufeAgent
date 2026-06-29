import httpx

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai import ModelSettings
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.models import create_async_http_client

from redlotus.config.app_config import apply_thinking_config, get_env
from redlotus.infra.shared_http import get_client


def create_model(model_name: str, parameter: dict):
    api_key = get_env("API_KEY", warn=False) or None
    api_base = get_env("BASE_URL", warn=False) or None
    profile_name = model_name.rsplit("/", 1)[-1]
    http_client = get_client(
        "model",
        lambda: create_async_http_client(timeout=300, connect=10),
    )
    param = apply_thinking_config(parameter, model_name=model_name)

    if 'gemini' in model_name:
        provider = GoogleProvider(base_url=api_base, api_key=api_key, http_client=http_client)
        profile = google_model_profile(profile_name)
        return GoogleModel(model_name, provider=provider, profile=profile, settings=ModelSettings(**param))

    if 'claude' in model_name:
        provider = AnthropicProvider(base_url=api_base, api_key=api_key, http_client=http_client)
        profile = anthropic_model_profile(model_name)
        return AnthropicModel(model_name, provider=provider, profile=profile, settings=ModelSettings(**param))

    provider = OpenAIProvider(base_url=api_base, api_key=api_key, http_client=http_client)
    if "deepseek" in model_name.lower():
        profile = deepseek_model_profile(profile_name)
    else:
        profile = OpenAIProvider.model_profile(profile_name)
    return OpenAIChatModel(model_name, provider=provider, profile=profile, settings=ModelSettings(**param))

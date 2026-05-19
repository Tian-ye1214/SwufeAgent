"""记忆模块兼容入口；实现已拆分至 tools.memory 子包。"""
from tools.memory import (
    ChatHistory,
    LongTermMemory,
    ShortTermMemory,
    UserMessage,
    long_term_memory_consolidation_params,
    merge_looks_like_unrelated_rewrite,
    message_has_user_prompt,
    pack_messages_to_chunk_texts,
    pydantic_messages_to_text,
    split_messages_into_turns,
    turn_has_agent_content,
    turn_texts_from_messages,
    user_message_from_cli_input,
    user_message_from_text,
)

# 保留旧版下划线前缀私有名，供既有调用方与测试使用
_long_term_memory_consolidation_params = long_term_memory_consolidation_params
_merge_looks_like_unrelated_rewrite = merge_looks_like_unrelated_rewrite
_pydantic_messages_to_text = pydantic_messages_to_text
_message_has_user_prompt = message_has_user_prompt
_turn_has_agent_content = turn_has_agent_content
_split_messages_into_turns = split_messages_into_turns
_turn_texts_from_messages = turn_texts_from_messages
_pack_messages_to_chunk_texts = pack_messages_to_chunk_texts

__all__ = [
    "ChatHistory",
    "LongTermMemory",
    "ShortTermMemory",
    "UserMessage",
    "user_message_from_cli_input",
    "user_message_from_text",
    "_long_term_memory_consolidation_params",
    "_merge_looks_like_unrelated_rewrite",
    "_pack_messages_to_chunk_texts",
    "_pydantic_messages_to_text",
    "_message_has_user_prompt",
    "_turn_has_agent_content",
    "_split_messages_into_turns",
    "_turn_texts_from_messages",
]

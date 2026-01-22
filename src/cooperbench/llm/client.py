"""
LLM client for making API calls to language models via litellm.

This module handles the interaction with various LLM providers through
a unified interface, including special handling for Qwen models with
context window management.
"""

import copy
import json
import re
from typing import Any

import litellm
from dotenv import load_dotenv
from litellm.caching.caching import Cache
from litellm.exceptions import ContextWindowExceededError
from litellm.utils import trim_messages

load_dotenv()
litellm.cache = Cache()


def _sanitize_special_tokens(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove special tokens that break tiktoken counting."""
    disallowed = ("<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|endofprompt|>", "<|fim_prefix|>")

    def clean_text(text: str) -> str:
        for marker in disallowed:
            if marker in text:
                text = text.replace(marker, f"[{marker.strip('<|>')}]")
        return text

    def clean_content(content: Any) -> Any:
        if isinstance(content, str):
            return clean_text(content)
        if isinstance(content, list):
            cleaned_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                    part = dict(part)
                    part["text"] = clean_text(part["text"])
                cleaned_parts.append(part)
            return cleaned_parts
        return content

    cleaned: list[dict[str, Any]] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        cleaned_msg = copy.deepcopy(msg)
        if "content" in cleaned_msg:
            cleaned_msg["content"] = clean_content(cleaned_msg["content"])
        cleaned.append(cleaned_msg)
    return cleaned


async def call_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    max_input_tokens: int | None = None,
    return_full_response: bool = False,
    **kwargs: object,
) -> tuple[list[dict[str, Any]] | None, str] | dict[str, Any]:
    """Call the LLM with the given messages and tools.
    
    Args:
        messages: List of message dicts (role, content)
        tools: List of tool schemas for function calling
        model: Model identifier string
        max_input_tokens: Max tokens for input (auto-set for Qwen models)
        return_full_response: Whether to return full response dict
        **kwargs: Additional kwargs passed to litellm

    Returns:
        If return_full_response: dict with tool_calls, content, and message
        Otherwise: tuple of (tool_calls list or None, content string)
    """
    is_qwen = "qwen" in model.lower()
    
    if max_input_tokens is None and is_qwen:
        max_input_tokens = 28000
    
    if is_qwen:
        sanitized_messages = _sanitize_special_tokens(messages)
    else:
        sanitized_messages = messages

    if not is_qwen:
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized_messages,
            "tools": tools,
            "tool_choice": "auto",
            "caching": True,
            "num_retries": 3,
            **kwargs,
        }
        if "gemini" in model.lower():
            call_kwargs["reasoning_effort"] = "low"
        response = await litellm.acompletion(**call_kwargs)
    else:
        try:
            trimmed_messages = trim_messages(sanitized_messages, model, max_tokens=max_input_tokens)
        except ValueError:
            trimmed_messages = sanitized_messages
        
        max_retries = 2
        current_max_tokens = max_input_tokens
        
        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=trimmed_messages,
                    tools=tools,
                    tool_choice="auto",
                    caching=True,
                    num_retries=3,
                    **kwargs,
                )
                break
            except ContextWindowExceededError:
                if attempt < max_retries:
                    current_max_tokens = int(current_max_tokens * 0.8)
                    print(f"[WARNING] Context window exceeded, retrying with max_tokens={current_max_tokens}")
                    try:
                        trimmed_messages = trim_messages(sanitized_messages, model, max_tokens=current_max_tokens)
                    except ValueError:
                        if len(sanitized_messages) > 3:
                            trimmed_messages = sanitized_messages[:2] + sanitized_messages[-2:]
                            print("[WARNING] Fallback: keeping only system, prompt, and last 2 messages")
                        else:
                            trimmed_messages = sanitized_messages
                else:
                    raise

    if not response or not response.choices:
        if return_full_response:
            return {"tool_calls": None, "content": "", "message": None}
        return None, ""

    message = response.choices[0].message
    content = message.content or ""

    tool_calls = None
    
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_calls = []
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            tool_call_dict: dict[str, Any] = {
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": arguments,
                },
            }
            
            if hasattr(tool_call, "provider_specific_fields") and tool_call.provider_specific_fields:
                tool_call_dict["provider_specific_fields"] = tool_call.provider_specific_fields
            
            tool_calls.append(tool_call_dict)
    
    if not tool_calls and content and is_qwen:
        tool_calls = _parse_tool_calls_from_content(content)
        if tool_calls:
            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()

    if return_full_response:
        if hasattr(message, 'model_dump'):
            message_dict = message.model_dump(exclude_none=True)
        else:
            message_dict = message
        return {
            "tool_calls": tool_calls,
            "content": content,
            "message": message_dict,
        }
    return tool_calls, content


def _parse_tool_calls_from_content(content: str) -> list[dict[str, Any]] | None:
    """Parse tool calls from <tool_call>...</tool_call> format (Qwen fallback)."""
    tool_calls = []
    
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for idx, match in enumerate(matches):
        try:
            tool_data = json.loads(match.strip())
            tool_name = tool_data.get("name", "")
            tool_args = tool_data.get("arguments", {})
            
            if tool_name:
                tool_calls.append(
                    {
                        "id": f"call_{idx}_{tool_name}",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args,
                        },
                    }
                )
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse tool call from content: {e}")
            continue
    
    return tool_calls if tool_calls else None

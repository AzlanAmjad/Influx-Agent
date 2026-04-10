"""
Shared LLM helpers used by graph nodes.

Centralises ChatOllama instantiation, JSON extraction from model
responses, and common message-list utilities so individual nodes stay
thin and consistent.
"""

import json

from langchain_ollama import ChatOllama

from app.src.core.config import settings


def get_llm(model: str, *, json_mode: bool = True) -> ChatOllama:
    """Return a deterministic ChatOllama instance.

    When *json_mode* is ``True`` (default) the model is instructed to
    return JSON, which is appropriate for structured-output nodes.  Set
    to ``False`` for free-form text generation (e.g. summarisation).
    """
    kwargs: dict = {
        "base_url": settings.ollama_host,
        "model": model,
        "temperature": 0,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def _find_balanced_braces(text: str) -> str | None:
    """
    Extract the first balanced ``{…}`` block from *text*, correctly
    skipping over JSON string literals (including escaped quotes).

    Returns ``None`` when no balanced object can be found – this handles
    the case where the LLM emits truncated or malformed JSON with
    unclosed braces far better than a greedy ``{.*}`` regex.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_json(content: str | dict) -> dict:
    """
    Best-effort JSON extraction from an Ollama response.

    Strategy (in order):
      1. ``content`` is already a dict → return directly.
      2. Whole string parses as JSON → return.
      3. Find the first *balanced* ``{…}`` block and parse that.

    Raises ``ValueError`` when no valid JSON can be recovered.
    """
    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        raise ValueError("Response content is not a string or dict")

    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    balanced = _find_balanced_braces(text)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

    raise ValueError("No JSON object found in LLM response")


def last_user_message(messages: list[dict]) -> str:
    """Return the content of the most recent user message, or ``""``."""
    return next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

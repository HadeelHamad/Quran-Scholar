"""Parse Tafsir MCP tool responses into Python dicts/lists."""

from __future__ import annotations

import json
from typing import Any


def mcp_payload(result: Any) -> Any:
    """Unwrap MCP tools/call result to structured data when possible."""
    if result is None:
        return None
    if isinstance(result, (dict, list)):
        if isinstance(result, dict):
            structured = result.get("structuredContent")
            if structured is not None:
                if isinstance(structured, dict) and "result" in structured:
                    return structured["result"]
                return structured
            content = result.get("content")
            if isinstance(content, list):
                items: list[Any] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text")
                    if text is None:
                        continue
                    try:
                        items.append(json.loads(text))
                    except (json.JSONDecodeError, TypeError):
                        items.append({"text": text})
                if len(items) == 1:
                    return items[0]
                return items
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"text": result}
    return result


def as_list(payload: Any) -> list[dict[str, Any]]:
    """Normalize a payload into a list of dict records."""
    data = mcp_payload(payload)
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("results", "ayahs", "verses", "hits", "items", "tafsirs"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    return []

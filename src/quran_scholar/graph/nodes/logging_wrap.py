"""Wrap graph nodes so each logs the dict it returns into ResearchState."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("quran_scholar.graph.nodes")

_MAX_STR = 240
_MAX_LIST_PREVIEW = 3


def _ensure_node_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _preview(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return _preview(value.model_dump(mode="json"))
    if isinstance(value, str):
        if len(value) <= _MAX_STR:
            return value
        return f"{value[:_MAX_STR]}… ({len(value)} chars)"
    if isinstance(value, dict):
        return {k: _preview(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = list(value)
        preview = [_preview(x) for x in items[:_MAX_LIST_PREVIEW]]
        if len(items) > _MAX_LIST_PREVIEW:
            preview.append(f"… +{len(items) - _MAX_LIST_PREVIEW} more")
        return {"count": len(items), "items": preview}
    if hasattr(value, "value"):  # Enum
        return getattr(value, "value", str(value))
    text = str(value)
    if len(text) <= _MAX_STR:
        return text
    return f"{text[:_MAX_STR]}… ({len(text)} chars)"


def summarize_state_update(updates: dict[str, Any] | None) -> dict[str, Any]:
    if not updates:
        return {}
    return {key: _preview(val) for key, val in updates.items()}


def with_state_logging(
    name: str, fn: Callable[[Any], dict[str, Any]]
) -> Callable[[Any], dict[str, Any]]:
    """Return a node that logs its state update after running ``fn``."""

    def wrapped(state: Any) -> dict[str, Any]:
        _ensure_node_logging()
        updates = fn(state) or {}
        summary = summarize_state_update(updates)
        try:
            body = json.dumps(summary, ensure_ascii=False, default=str, indent=2)
        except TypeError:
            body = str(summary)
        logger.info("[%s] state update →\n%s", name, body)
        return updates

    wrapped.__name__ = getattr(fn, "__name__", name)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped

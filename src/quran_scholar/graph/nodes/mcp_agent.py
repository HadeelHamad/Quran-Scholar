"""Run researcher agents via LangChain ``create_agent`` + Tafsir MCP tools."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from quran_scholar.graph.nodes.llm import get_llm
from quran_scholar.mcp.client import get_tafsir_mcp_tools
from quran_scholar.mcp.toolsets import ResearcherRole

T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolCallRecord:
    name: str
    content: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearcherAgentResult:
    structured: BaseModel | None
    tool_calls: list[ToolCallRecord]
    warnings: list[str] = field(default_factory=list)
    raw_messages: list[Any] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [t.name for t in self.tool_calls]


def has_llm_credentials() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("your_")


def run_researcher_agent(
    *,
    role: ResearcherRole,
    system_prompt: str,
    user_message: str,
    response_format: type[T] | None = None,
    name: str | None = None,
) -> ResearcherAgentResult:
    """
    create_agent loop: the model chooses among role-scoped MCP tools.

    Evidence should be derived from ``tool_calls`` contents (grounded), not from
    free-form model prose.
    """
    warnings: list[str] = []
    if not has_llm_credentials():
        return ResearcherAgentResult(
            structured=None,
            tool_calls=[],
            warnings=["mcp_agent: no OPENAI_API_KEY — cannot run create_agent"],
        )

    client, tools = get_tafsir_mcp_tools(role=role)
    if not tools:
        client.close()
        return ResearcherAgentResult(
            structured=None,
            tool_calls=[],
            warnings=[f"mcp_agent: no MCP tools available for role={role}"],
        )

    try:
        agent = create_agent(
            get_llm(),
            tools,
            system_prompt=system_prompt,
            response_format=response_format,
            name=name or f"{role}_researcher",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
    except Exception as exc:
        warnings.append(f"mcp_agent: create_agent failed for role={role}: {exc}")
        return ResearcherAgentResult(
            structured=None, tool_calls=[], warnings=warnings
        )
    finally:
        client.close()

    messages = list(result.get("messages") or [])
    tool_calls = _extract_tool_calls(messages)
    structured = result.get("structured_response")
    if response_format is not None and structured is not None:
        if not isinstance(structured, response_format):
            try:
                structured = response_format.model_validate(structured)
            except Exception as exc:
                warnings.append(f"mcp_agent: structured_response invalid: {exc}")
                structured = None

    if not tool_calls:
        warnings.append(
            f"mcp_agent: agent finished without calling tools (role={role})"
        )

    return ResearcherAgentResult(
        structured=structured,
        tool_calls=tool_calls,
        warnings=warnings,
        raw_messages=messages,
    )


def _extract_tool_calls(messages: list[Any]) -> list[ToolCallRecord]:
    """Pair AI tool_call args with ToolMessage results."""
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    out: list[ToolCallRecord] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tid = str(tc.get("id") or "")
                name = str(tc.get("name") or "unknown")
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                if tid:
                    pending[tid] = (name, args)
        elif isinstance(msg, ToolMessage):
            tid = str(getattr(msg, "tool_call_id", "") or "")
            name_from_msg = getattr(msg, "name", None)
            name, args = pending.get(tid, ("unknown", {}))
            if name_from_msg:
                name = str(name_from_msg)
            content = msg.content if isinstance(msg.content, str) else json.dumps(
                msg.content, ensure_ascii=False
            )
            out.append(ToolCallRecord(name=name, content=content, args=args))
    return out


def parse_tool_json(content: str) -> Any:
    """Parse tool return string (JSON) into Python objects."""
    text = (content or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def tool_had_mcp_error(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("mcp_error") is True

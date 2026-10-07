"""Safe MCP calls — failures become outcomes, never crash the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.toolsets import ToolNotAllowedError


@dataclass
class MCPCallResult:
    """ok=transport succeeded; failed=MCP error (never treat as 'no evidence')."""

    tool: str
    ok: bool = False
    failed: bool = False
    empty: bool = False
    data: Any = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def safe_call_tool(
    client: Any,
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> MCPCallResult:
    where = label or tool
    try:
        data = client.call_tool(tool, arguments or {})
    except (MCPError, ToolNotAllowedError, httpx.HTTPError, Exception) as exc:
        kind = "MCP error" if isinstance(exc, MCPError) else (
            "network" if isinstance(exc, httpx.HTTPError) else "unexpected"
        )
        msg = f"{where} retrieval failed ({kind} — not 'no evidence'): {exc}"
        return MCPCallResult(
            tool=tool, failed=True, error=str(exc), warnings=[msg]
        )

    empty = data is None or data in ({}, [], "")
    return MCPCallResult(tool=tool, ok=True, empty=empty, data=data)


def mark_empty(result: MCPCallResult, message: str) -> list[str]:
    """Append a NO_EVIDENCE warning only when the call did not fail."""
    if result.failed:
        return list(result.warnings)
    result.empty = True
    result.warnings.append(message)
    return list(result.warnings)

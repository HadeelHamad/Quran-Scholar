"""Safe MCP call wrappers — failures become outcomes, never crash the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from quran_scholar.mcp.errors import MCPError, MCPNetworkError, MCPProtocolError
from quran_scholar.mcp.toolsets import ToolNotAllowedError


class _MCPCaller(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


@dataclass
class MCPCallResult:
    """
    Distinguishes successful empty evidence from MCP failure.

    - ok + empty=False → data retrieved
    - ok + empty=True  → request succeeded; no evidence exists for this query
    - failed=True      → MCP request failed (do NOT treat as "no evidence")
    """

    tool: str
    ok: bool
    empty: bool = False
    failed: bool = False
    data: Any = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def succeeded_with_data(self) -> bool:
        return self.ok and not self.empty and not self.failed


def safe_call_tool(
    client: _MCPCaller,
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> MCPCallResult:
    """
    Wrap an MCP tool call.

    Network / protocol errors become ``failed=True`` warnings — they must never
    be interpreted as "no evidence exists."
    """
    where = label or tool
    try:
        data = client.call_tool(tool, arguments or {})
    except ToolNotAllowedError as exc:
        # Programming error: still don't crash the graph
        msg = f"MCP tool not allowed for this researcher ({where}): {exc}"
        return MCPCallResult(
            tool=tool,
            ok=False,
            failed=True,
            error=str(exc),
            warnings=[msg],
        )
    except MCPError as exc:
        msg = f"{where} retrieval failed (MCP error — not 'no evidence'): {exc}"
        return MCPCallResult(
            tool=tool,
            ok=False,
            failed=True,
            error=str(exc),
            warnings=[msg],
        )
    except httpx.HTTPError as exc:
        msg = f"{where} retrieval failed (network — not 'no evidence'): {exc}"
        return MCPCallResult(
            tool=tool,
            ok=False,
            failed=True,
            error=str(exc),
            warnings=[msg],
        )
    except Exception as exc:
        # Catch-all so one bad MCP call cannot crash LangGraph
        msg = f"{where} retrieval failed (unexpected — not 'no evidence'): {exc}"
        return MCPCallResult(
            tool=tool,
            ok=False,
            failed=True,
            error=str(exc),
            warnings=[msg],
        )

    # Successful transport — decide empty vs data at call site usually;
    # here we only flag obviously empty payloads.
    empty = data is None or data == {} or data == [] or data == ""
    return MCPCallResult(tool=tool, ok=True, empty=empty, data=data)


def mark_empty(
    result: MCPCallResult,
    *,
    message: str,
) -> MCPCallResult:
    """Annotate a successful call that returned no usable evidence."""
    if result.failed:
        # Never convert a failure into an empty-evidence note
        return result
    result.empty = True
    result.warnings.append(message)
    return result

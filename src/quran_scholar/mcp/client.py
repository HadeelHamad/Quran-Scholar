"""Tafsir MCP client over Streamable HTTP (remote) or documented local uvx mode.

Uses httpx so we do not require fastmcp/cryptography native builds.
Endpoint docs: https://tafsirmcp.netlify.app/ · https://mcp.tafsir.net/mcp
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

DEFAULT_TAFSIR_MCP_URL = "https://mcp.tafsir.net/mcp"

# Core tools used by Quran Scholar research nodes
PRIMARY_TOOLS = frozenset(
    {
        "fetch_ayah",
        "fetch_tafsir",
        "fetch_nuzool_reason",
        "search_quran_text",
        "search_in_tafsir",
        "analyze_word",
        "find_root_occurrences",
        "get_root_stats",
        "fetch_surah_info",
        "get_qeraat_variants",
    }
)


class TafsirMCPError(RuntimeError):
    """Raised when the Tafsir MCP HTTP session fails."""


class TafsirMCPClient:
    """Minimal JSON-RPC client for MCP Streamable HTTP."""

    def __init__(self, url: str | None = None, *, timeout: float = 60.0) -> None:
        self.url = url or os.getenv("TAFSIR_MCP_URL", DEFAULT_TAFSIR_MCP_URL)
        self.timeout = timeout
        self._session_id: str | None = None
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TafsirMCPClient:
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        session = response.headers.get("mcp-session-id") or response.headers.get(
            "Mcp-Session-Id"
        )
        if session:
            self._session_id = session

        if response.status_code >= 400:
            raise TafsirMCPError(
                f"MCP HTTP {response.status_code}: {response.text[:500]}"
            )

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse(response.text)
        if not response.content:
            return {}
        return response.json()

    @staticmethod
    def _parse_sse(body: str) -> dict[str, Any]:
        """Extract the last JSON-RPC payload from an SSE stream body."""
        data_lines: list[str] = []
        last_payload: dict[str, Any] | None = None
        for line in body.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.strip() == "" and data_lines:
                raw = "\n".join(data_lines)
                data_lines = []
                try:
                    last_payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
        if data_lines:
            try:
                last_payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                pass
        if last_payload is None:
            raise TafsirMCPError("No JSON payload in MCP SSE response")
        return last_payload

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        response = self._client.post(
            self.url, headers=self._headers(), content=json.dumps(payload)
        )
        data = self._parse_response(response)
        if "error" in data and data["error"]:
            raise TafsirMCPError(str(data["error"]))
        return data.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        response = self._client.post(
            self.url, headers=self._headers(), content=json.dumps(payload)
        )
        # notifications may be 202/empty
        if response.status_code >= 400:
            raise TafsirMCPError(
                f"MCP notify failed {response.status_code}: {response.text[:300]}"
            )
        sid = response.headers.get("mcp-session-id") or response.headers.get(
            "Mcp-Session-Id"
        )
        if sid:
            self._session_id = sid

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "quran-scholar", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized")
        return result or {}

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list")
        return list((result or {}).get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        return result


def _schema_to_args_model(tool_name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
    """Build a loose Pydantic args model from an MCP JSON schema."""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    fields: dict[str, Any] = {}
    for key, spec in props.items():
        annotation: Any = Any
        typ = spec.get("type")
        if typ == "string":
            annotation = str
        elif typ == "integer":
            annotation = int
        elif typ == "number":
            annotation = float
        elif typ == "boolean":
            annotation = bool
        elif typ == "array":
            annotation = list
        elif typ == "object":
            annotation = dict
        default = ... if key in required else None
        fields[key] = (
            annotation | None if default is None else annotation,
            Field(default=default, description=spec.get("description")),
        )
    if not fields:
        fields["payload"] = (dict[str, Any] | None, Field(default=None))
    return create_model(f"{tool_name}_Args", **fields)


def _normalize_tool_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    # MCP tools/call often returns {content: [{type:text, text:...}]}
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            return "\n".join(parts)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def get_tafsir_mcp_tools(
    *,
    url: str | None = None,
    primary_only: bool = True,
) -> tuple[TafsirMCPClient, list[StructuredTool]]:
    """
    Open a Tafsir MCP session and wrap tools as LangChain StructuredTools.

    Caller owns the client lifecycle (call ``client.close()`` when done).
    """
    mode = os.getenv("TAFSIR_MCP_MODE", "http").strip().lower()
    if mode == "local":
        raise TafsirMCPError(
            "TAFSIR_MCP_MODE=local requires a running MCP stdio bridge. "
            "Use TAFSIR_MCP_MODE=http (default) against "
            f"{DEFAULT_TAFSIR_MCP_URL}, or set TAFSIR_MCP_URL to your proxy."
        )

    client = TafsirMCPClient(url=url)
    client.initialize()
    catalog = client.list_tools()

    tools: list[StructuredTool] = []
    for meta in catalog:
        name = meta.get("name")
        if not name:
            continue
        if primary_only and name not in PRIMARY_TOOLS:
            continue
        description = meta.get("description") or f"Tafsir MCP tool: {name}"
        args_model = _schema_to_args_model(name, meta.get("inputSchema"))

        def _make_coroutine(tool_name: str):
            def _call(**kwargs: Any) -> str:
                cleaned = {k: v for k, v in kwargs.items() if v is not None}
                if "payload" in cleaned and len(cleaned) == 1:
                    cleaned = cleaned["payload"] or {}
                result = client.call_tool(tool_name, cleaned)
                return _normalize_tool_result(result)

            return _call

        tools.append(
            StructuredTool.from_function(
                func=_make_coroutine(name),
                name=name,
                description=description,
                args_schema=args_model,
            )
        )

    return client, tools


def tafsir_mcp_adapter():
    """Backward-compatible name: prefer get_tafsir_mcp_tools()."""
    return get_tafsir_mcp_tools()

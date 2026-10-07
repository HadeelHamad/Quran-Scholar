# Quran Scholar

A Multi-Agent Quranic Research and Tafsir System.

Orchestrates LangGraph agents with [Tafsir MCP](https://tafsirmcp.netlify.app/) for verified Quranic text, classical tafsir, linguistic analysis, and asbab al-nuzool — so answers stay cited instead of hallucinated.

## Stack

| Layer | Role |
| --- | --- |
| **LangGraph** | Workflow orchestration (non-sequential: planner, parallel research, verify/retry) |
| **LLM agents** | Planning, comparison, analysis, verification decisions |
| **Tafsir MCP** | Authenticated retrieval (`fetch_ayah`, `fetch_tafsir`, `search_quran_text`, …) |
| **Pydantic** | Structured plans, evidence, claims, verification |
| **Deterministic Python** | Validation, routing, iteration counters, state updates |

## Project layout

```
src/quran_scholar/
├── models.py       # Pydantic evidence / plan / claim schemas
├── state.py        # Shared ResearchState for the graph
├── mcp/            # Tafsir MCP client adapter
├── agents/         # LLM-backed agents
├── nodes/          # Graph nodes (agents + deterministic)
└── graph/          # StateGraph wiring
```

## Setup

Requires Python **≥ 3.12** (Tafsir MCP dependency).

```bash
cp .env.example .env   # add OPENAI_API_KEY (and optional LangSmith / MCP settings)
uv sync
```

### Tafsir MCP

Default: remote Streamable HTTP endpoint (no local DB install required):

```env
TAFSIR_MCP_URL=https://mcp.tafsir.net/mcp
```

Tools are wrapped in `src/quran_scholar/mcp/client.py` as LangChain tools
(`fetch_ayah`, `fetch_tafsir`, `search_quran_text`, …).

Optional offline server: `uvx tafsir-mcp` (needs working Xcode CLT on macOS if
native wheels fail). Point a local proxy at `TAFSIR_MCP_URL` if you bridge stdio→HTTP.

## Design notes

- One shared `ResearchState`; nodes return **only** fields they change.
- Not every node is an agent — use LLMs for interpretation/planning; use Python for validation/routing.
- Research iterates until verification passes or `max_research_iterations` is hit.
- Evidence store lives in graph state (`evidence_items` + `claims`).

## Attribution

Quranic data via Tafsir MCP is **CC BY 4.0** — attribute [Tafsir Center for Quranic Studies](https://tafsir.net).

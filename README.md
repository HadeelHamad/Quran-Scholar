# Quran Scholar

A Multi-Agent Quranic Research and Tafsir System.

Orchestrates LangGraph agents with [Tafsir MCP](https://tafsirmcp.netlify.app/) for verified Quranic text, classical tafsir, linguistic analysis, and asbab al-nuzool — so answers stay cited instead of hallucinated.

**Input:** Users ask research questions in **Modern Standard Arabic** (e.g. `ما تفسير آية الكرسي؟`, `سورة ٢ آية ٢٥٥`, or `2:255`). The final report is Arabic; `language` defaults to `ar` in `initial_research_state`.

## Stack

| Layer | Role |
| --- | --- |
| **LangGraph** | Workflow orchestration (non-sequential: planner, parallel research, verify/retry) |
| **LLM agents** | Planning, comparison, analysis, verification; researchers use `create_agent` + MCP tools |
| **Tafsir MCP** | Tools bound per role (`fetch_ayah`, `search_quran_text`, `fetch_tafsir`, `search_in_tafsir`, …) |
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
├── graph/          # StateGraph wiring
└── web/            # FastAPI UI (Arabic question → report)
```

## Web UI

Run the local server (loads `.env`, compiles the graph once at startup):

```bash
uv sync
uv run quran-scholar-web
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), type your question in Arabic, and submit. The page shows the final Arabic report, optional trace log, and any warnings.

Options: `uv run quran-scholar-web --port 8080 --reload`

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


## System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Web UI (FastAPI)  ← Arabic question → final Arabic report      │
└────────────────────────────┬────────────────────────────────────┘
                             │ invoke(ResearchState)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph StateGraph                        │
│                                                                 │
│  START → Planner → Research Manager ──┬──► Quran Researcher ─┐  │
│                    ▲                  ├──► Tafsir Researcher ┼──┤
│                    │                  ├──► Linguistic Res.   ┘  │
│                    │                  └──► Context Researcher ─┤
│                    │                                            │
│                    └──────── Gap Analyzer ◄─────────────────────┘
│                              │                                  │
│                 insufficient │ sufficient                       │
│                 (more work)  ▼                                  │
│                              Tafsir Comparator                  │
│                                       │                         │
│                              Claim Extractor                    │
│                                       │                         │
│                              Evidence Verifier                  │
│                         ┌─────┴──────┐                          │
│                    retry│            │passed / max iters        │
│                         ▼            ▼                          │
│                  Gap Analyzer   Report Generator → END          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              Tafsir MCP (HTTP) — role-scoped tools via create_agent
```

**Shared state:** one `ResearchState` (`state.py`). Nodes return only fields they change. Evidence lists use append reducers so retries accumulate rather than wipe prior work.

**External services:** LLM (OpenAI-compatible / OpenRouter) for agents; Tafsir MCP for Quran/tafsir/linguistic/nuzool data.

Wired in `src/quran_scholar/graph/builder.py`.

Compiled graph diagram (exported from LangGraph):

![Quran Scholar LangGraph](docs/quran_scholar_graph.png)

Regenerate with:

```bash
uv run python -c "
from pathlib import Path
from quran_scholar.graph import build_graph
Path('docs/quran_scholar_graph.png').write_bytes(
    build_graph().get_graph().draw_mermaid_png()
)
"
```

### Node responsibilities

| Node | Responsibility |
| --- | --- |
| **Planner** | Turn the Arabic question into a **research plan only** (focus, tasks, needed evidence types). Does not answer the question or quote tafsir. |
| **Research Manager** | Supervisor: pick the next action(s) from the plan (and gaps). Can fan out **parallel** researchers when tasks are independent. |
| **Quran Researcher** | Find relevant ayahs (`create_agent` + `fetch_ayah` / `search_quran_text`). Writes `discovered_verses` and relevance-filtered `selected_verses`. |
| **Tafsir Researcher** | Retrieve classical commentary for selected verses (`fetch_tafsir` and/or `search_in_tafsir`). Stores raw attributed excerpts — no LLM rewrite of tafsir. |
| **Linguistic Researcher** | Optional word/root study (`analyze_word`, `get_root_stats`, `find_root_occurrences`) when the plan needs it. |
| **Context Researcher** | Fetch asbab al-nuzool (`fetch_nuzool_reason`) with status FOUND / NOT_AVAILABLE / ERROR. |
| **Gap Analyzer** | Check whether collected evidence is enough for the plan; mark gaps and send the run back to the manager or onward. |
| **Tafsir Comparator** | Compare tafsir sources on the same verse(s): agreements, differences, open questions. |
| **Claim Extractor** | Turn verified materials into auditable **claims**, each tied to `evidence_ids` (no free-floating assertions). |
| **Evidence Verifier** | Check claims against evidence; pass → report, fail → loop via gap analyzer (until max iterations). |
| **Report Generator** | Write the final **Arabic** report from verified claims/evidence/citations only — no invented Quranic facts. |

### Routing (short)

- **After Research Manager:** one researcher, several in parallel, gap analysis, comparison, verification, or finish (report).
- **After Gap Analyzer:** back to manager if gaps/pending tasks; else tafsir comparator.
- **After Evidence Verifier:** report if passed or iteration limit hit; else gap analyzer for another research wave.

Smoke test (logs a live agent trace via `quran_scholar.trace`, then the report):

```bash
uv run python -c "
import logging
from quran_scholar.graph import build_graph
from quran_scholar.state import initial_research_state
from quran_scholar.trace import format_trace_log, logger as trace_logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
out = build_graph().invoke(initial_research_state('ما تفسير آية الكرسي؟'))
trace_logger.info(format_trace_log(out.get('trace_log')))
trace_logger.info(out.get('final_report') or '')
"
```

Trace lines look like:

```
[Planner] Creating research plan...
[ResearchManager] Selecting Quran research...
[QuranResearcher] Searching Quran...
[QuranResearcher] Found 14 candidate verses.
[QuranResearcher] Selected 8 relevant verses.
...
[ReportGenerator] Generating final report.
```

## Design notes

- One shared `ResearchState`; nodes return **only** fields they change.
- Append reducers on evidence/claims/findings so iterations cannot wipe prior work.
- Not every node is an agent — use LLMs for interpretation/planning; use Python for validation/routing.
- Research iterates until verification passes or `max_research_iterations` is hit.

## Attribution

Quranic data via Tafsir MCP is **CC BY 4.0** — attribute [Tafsir Center for Quranic Studies](https://tafsir.net).

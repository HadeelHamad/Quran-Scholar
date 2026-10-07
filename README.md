# Quran Scholar

A multi-agent system for **any Quran-related question** — not only tafsir.

The Planner builds a minimal investigation plan; specialized researchers use
[Tafsir MCP](https://tafsirmcp.netlify.app/) tools (as needed) for Quran text,
search, surah/overview/stats/qira'at, classical tafsir, linguistics, and asbab
al-nuzool. Answers stay cited instead of hallucinated.

**Examples:** thematic verses · verse lookup · tafsir / mufassir comparison ·
word roots · reasons of revelation · surah info / statistics / qira'at.

**Input:** questions in **Modern Standard Arabic**. Final report is Arabic;
`language` defaults to `ar`.

## Stack

| Layer | Role |
| --- | --- |
| **LangGraph** | Workflow orchestration (non-sequential: planner, parallel research, verify/retry) |
| **Tool agents (`*_agent`)** | Researchers only: `create_agent` + MCP tools |
| **Other LLM modules** | Planner, manager, gap, comparator, claims, verifier, report — LLM/heuristics, **no** tools |
| **Tafsir MCP** | Role toolsets (Quran meta + text, tafsir, linguistic, nuzool) — planner only schedules what the question needs |
| **Pydantic** | Structured plans, evidence, claims, verification |
| **Deterministic Python** | Validation, routing, iteration counters, state updates |

## Project layout

```
src/quran_scholar/
├── models.py       # Pydantic evidence / plan / claim schemas
├── state.py        # Shared ResearchState for the graph
├── mcp/            # Tafsir MCP client adapter
├── agents/         # Tool agents (*_agent) + LLM helpers (no tools)
├── nodes/          # Graph node wrappers + routing
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
| **Quran Researcher** | Text + meta MCP tools (`fetch_ayah`, `search_quran_text`, surah info, overview, stats, qira'at, …). |
| **Tafsir Researcher** | Classical commentary **when planned** (`fetch_tafsir`, `search_in_tafsir`, list sources). |
| **Linguistic Researcher** | Word/root study **when planned**. |
| **Context Researcher** | Asbab al-nuzool / source lists **when planned**. |
| **Gap Analyzer** | Check whether collected evidence is enough for the plan; mark gaps and send the run back to the manager or onward. |
| **Tafsir Comparator** | Compare tafsir sources on the same verse(s): agreements, differences, open questions. |
| **Claim Extractor** | Turn verified materials into auditable **claims**, each tied to `evidence_ids` (no free-floating assertions). |
| **Evidence Verifier** | Check claims against evidence; pass → report, fail → loop via gap analyzer (until max iterations). |
| **Report Generator** | Final **Arabic Q&A**: direct **الإجابة** + **الأدلة** (citations/excerpts) from collected evidence only. |

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
- Only researchers are named `*_agent` (LLM + tools). Planner/manager/analysis/report use LLM or heuristics without tools.
- Research iterates until verification passes or `max_research_iterations` is hit.

## Attribution

Quranic data via Tafsir MCP is **CC BY 4.0** — attribute [Tafsir Center for Quranic Studies](https://tafsir.net).

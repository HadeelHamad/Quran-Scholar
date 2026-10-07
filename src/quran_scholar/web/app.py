"""FastAPI server: Arabic question form → research graph → final report."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Quran Scholar", version="0.1.0")
_graph: Any = None


@app.on_event("startup")
def _load_graph() -> None:
    global _graph
    from quran_scholar.graph import build_graph

    _graph = build_graph()


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ResearchResponse(BaseModel):
    report: str | None = None
    trace_log: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    research_complete: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/research", response_model=ResearchResponse)
async def run_research(body: ResearchRequest) -> ResearchResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="السؤال فارغ")

    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    from quran_scholar.state import initial_research_state

    loop = asyncio.get_running_loop()

    def _invoke() -> dict[str, Any]:
        return _graph.invoke(initial_research_state(question))

    try:
        out = await loop.run_in_executor(None, _invoke)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"فشل البحث: {exc}",
        ) from exc

    return ResearchResponse(
        report=out.get("final_report"),
        trace_log=list(out.get("trace_log") or []),
        errors=list(out.get("errors") or []),
        warnings=list(out.get("warnings") or []),
        research_complete=bool(out.get("research_complete")),
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

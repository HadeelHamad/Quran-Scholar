"""Web UI for Quran Scholar."""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Quran Scholar web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "quran_scholar.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

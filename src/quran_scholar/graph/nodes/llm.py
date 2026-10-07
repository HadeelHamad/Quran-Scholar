"""Shared LLM configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(*, temperature: float = 0) -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "cohere/north-mini-code:free")
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)

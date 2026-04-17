from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Any, Optional

import requests

load_dotenv(override=True)
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


class OllamaError(RuntimeError):
    pass


def chat_with_ollama(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    *,
    base_url: str | None = DEFAULT_OLLAMA_BASE_URL,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: float = 120.0,
    options: Optional[dict[str, Any]] = None,
) -> str:
    """
    Send chat messages to Ollama and return only the assistant reply text.

    Parameters
    ----------
    messages:
        Chat messages in Ollama/OpenAI style:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

    model:
        Ollama model name, for example, "gemma4:e4b", "llama3.1:8b", etc.
        If omitted, this function tries the OLLAMA_MODEL environment variable.

    base_url:
        Ollama server URL. Default: http://localhost:11434

    temperature, top_p:
        Convenience options. They will be added to Ollama's option object.

    timeout:
        Request timeout in seconds.

    options:
        Extra Ollama options. Example:
        {"num_ctx": 4096, "seed": 42}
    """
    final_model = model or DEFAULT_OLLAMA_MODEL
    if not final_model:
        raise ValueError(
            "No model was provided. Pass model=... or set the OLLAMA_MODEL environment variable."
        )

    payload: dict[str, Any] = {
        "model": final_model,
        "messages": messages,
        "stream": False,
    }

    merged_options: dict[str, Any] = {}
    if options:
        merged_options.update(options)

    if temperature is not None:
        merged_options["temperature"] = temperature
    if top_p is not None:
        merged_options["top_p"] = top_p

    if merged_options:
        payload["options"] = merged_options

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise OllamaError(f"Failed to connect to Ollama: {exc}") from exc
    except ValueError as exc:
        raise OllamaError("Ollama returned a non-JSON response.") from exc

    try:
        return data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise OllamaError(f"Unexpected Ollama response format: {data}") from exc

from __future__ import annotations

import os
from functools import lru_cache
from dotenv import load_dotenv
from typing import Any, Optional

from openai import APIError, APITimeoutError, OpenAI

load_dotenv(override=True)
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class OllamaError(RuntimeError):
    pass


def _display_markdown_notebook(text: str, display_id: str | None = None) -> str | None:
    """
    Best-effort Markdown rendering in notebook environments.
    Returns a display_id when creating a new display handle, else None.
    """
    try:
        from IPython.display import Markdown, display, update_display
    except Exception:
        return None

    if display_id is None:
        handle = display(Markdown(text), display_id=True)
        return getattr(handle, "display_id", None)

    update_display(Markdown(text), display_id=display_id)
    return display_id


def _normalize_openai_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


@lru_cache(maxsize=8)
def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=_normalize_openai_base_url(base_url), api_key=api_key)


def chat_with_ollama(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    *,
    base_url: str | None = DEFAULT_OLLAMA_BASE_URL,
    api_key: str | None = DEFAULT_OLLAMA_API_KEY,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: float = 180.0,
    response_format: Optional[dict[str, Any]] = None,
    stream: bool = True,
    stream_to_stdout: bool = True,
    render_markdown: bool = True,
    options: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Send chat messages to Ollama and return assistant text.

    Parameters
    ----------
    messages:
        Chat messages in Ollama/OpenAI style:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

    model:
        Model name, for example, "gemma4:e4b", "llama3.1:8b", etc.
        If omitted, this function tries the OLLAMA_MODEL environment variable.

    base_url:
        OpenAI-compatible server URL. Default: http://localhost:11434

    api_key:
        API key for Ollama's OpenAI-compatible endpoint.
        For local Ollama, any non-empty value works.

    temperature, top_p:
        Convenience options. They will be added to Ollama's option object.

    timeout:
        Request timeout in seconds.

    options:
        Extra Ollama-native options sent through extra_body. Example:
        {"num_ctx": 4096, "seed": 42}

    response_format:
        OpenAI-compatible response format passthrough, for example:
        {"type": "json_object"}

    stream:
        If True, returns an iterable of streaming chunks.
        If False (default), returns a single assistant string.

    stream_to_stdout:
        Stream output control when stream=True and render_markdown=False:
        - True: print streamed tokens and return None
        - False: return raw stream iterator

    render_markdown:
        If True (default), render responses as Markdown in notebook environments.
        For stream=True, this enables live Markdown updates and returns None.
        This takes priority over stream_to_stdout.
    """
    final_model = model or DEFAULT_OLLAMA_MODEL
    if not final_model:
        raise ValueError(
            "No model was provided. Pass model=... or set the OLLAMA_MODEL environment variable."
        )

    merged_options: dict[str, Any] = {}
    if options:
        merged_options.update(options)

    if temperature is not None:
        merged_options["temperature"] = temperature
    if top_p is not None:
        merged_options["top_p"] = top_p

    if not base_url:
        raise ValueError(
            "No base_url was provided. Pass base_url=... or set OLLAMA_BASE_URL."
        )

    try:
        final_api_key = api_key or DEFAULT_OLLAMA_API_KEY
        client = _get_openai_client(base_url, final_api_key)

        completion = client.chat.completions.create(
            model=final_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            stream=stream,
            extra_body={"options": merged_options} if merged_options else None,
        )
    except (APIError, APITimeoutError) as exc:
        raise OllamaError(f"Failed to connect to Ollama: {exc}") from exc
    except Exception as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc

    if stream:
        if stream_to_stdout is False:
            return completion

        if render_markdown:
            streamed_text = ""
            display_id: str | None = _display_markdown_notebook("")
            for chunk in completion:
                delta = ""
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content or ""
                if delta:
                    streamed_text += delta
                    if display_id is not None:
                        _display_markdown_notebook(streamed_text, display_id=display_id)
                    else:
                        print(delta, end="", flush=True)
            if display_id is None:
                print()
            return None

        if stream_to_stdout is True:
            for chunk in completion:
                delta = ""
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content or ""
                if delta:
                    print(delta, end="", flush=True)
            print()
            return None

        return completion

    try:
        response_text = completion.choices[0].message.content or ""
        if render_markdown:
            display_id = _display_markdown_notebook(response_text)
            if display_id is None:
                print(response_text)
            return None
        return response_text
    except (AttributeError, IndexError, TypeError) as exc:
        raise OllamaError(
            f"Unexpected Ollama response format: {completion.model_dump()}"
        ) from exc

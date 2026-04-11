"""LLM client with schema validation and retry logic.

Supports three providers:
- **ollama** (local, default): http://localhost:11434
- **openrouter** (cloud, OpenAI-compatible): https://openrouter.ai/api/v1
- **huggingface** (cloud, OpenAI-compatible router): https://router.huggingface.co/v1

All three backends share the same retry + parse + validate pipeline.
A single dispatcher `call_llm()` routes based on the `provider` argument.
"""

import json
import time
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMConnectionError(Exception):
    """Raised when an LLM backend (local or cloud) is unreachable."""

    pass


class LLMValidationError(Exception):
    """Raised when LLM output fails schema validation after retries."""

    pass


# Backwards compatible aliases (old code imports these names)
OllamaConnectionError = LLMConnectionError
OllamaValidationError = LLMValidationError


# Supported provider → config
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "requires_key": False,
        "default_model": "qwen2.5:7b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "requires_key": True,
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
    },
}


def _parse_and_validate(
    raw_output: str, schema: type[T]
) -> tuple[T | None, str | None]:
    """Try to parse raw text into a validated schema instance.

    Returns:
        (instance, None) on success, (None, error_message) on failure.
    """
    try:
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = raw_output[json_start:json_end]
        else:
            json_str = raw_output

        parsed = json.loads(json_str)
        return schema(**parsed), None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, str(e)


def _build_correction_prompt(raw_output: str, error: str, schema: type[T]) -> str:
    """Build a focused correction prompt WITHOUT re-sending the transcript.

    On retry, the model already processed the transcript. Sending it again
    wastes tokens and confuses 7B models. Instead, show the previous output
    and the specific validation error.
    """
    return f"""Your previous response was invalid JSON. Fix the errors below.

[YOUR PREVIOUS OUTPUT]
{raw_output}

[VALIDATION ERRORS]
{error}

Fix these errors and return ONLY a corrected JSON object. No markdown, no explanation."""


def call_ollama(
    prompt: str,
    schema: type[T],
    model: str = "qwen2.5:7b",
    temperature: float = 0.55,
    max_retries: int = 2,
    timeout: int = 300,
) -> T:
    """Call Ollama (local) and validate response against Pydantic schema.

    Default timeout is 5min: a 7B model on Apple Silicon typically produces
    the full schema in 30-90s, but long transcripts + first-call cold start
    (model loading) can push a single request past the 2min mark.
    """
    base_url = PROVIDERS["ollama"]["base_url"]

    # Health check — cheap and catches the "Ollama not running" case early
    try:
        health = requests.get(f"{base_url}/api/tags", timeout=5)
        health.raise_for_status()
    except requests.RequestException:
        raise LLMConnectionError(
            "[llm] error: Ollama not responding at localhost:11434. "
            "Start Ollama with: `ollama serve` or `ollama run qwen2.5:7b`"
        )

    current_prompt = prompt
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            print(f"[llm] ollama {model}... (attempt {attempt + 1}/{max_retries + 1})")
            start_time = time.time()

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": current_prompt,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()

            duration = time.time() - start_time
            raw_output = response.json().get("response", "").strip()
            print(f"[llm] got response in {duration:.1f}s")

            result, error = _parse_and_validate(raw_output, schema)
            if result is not None:
                print("[llm] ✓ response valid")
                return result

            last_error = error
            if attempt < max_retries:
                current_prompt = _build_correction_prompt(raw_output, error, schema)
            else:
                raise LLMValidationError(
                    f"[llm] error: validation failed after {max_retries + 1} attempts. "
                    f"Last error: {last_error}\n\nRaw output:\n{raw_output}"
                )

        except requests.Timeout:
            raise LLMValidationError(
                f"[llm] error: timeout after {timeout}s. Try a smaller model."
            )
        except requests.RequestException as e:
            raise LLMConnectionError(f"[llm] error: {e}")

    raise LLMValidationError(
        f"[llm] error: unexpected state after {max_retries + 1} attempts"
    )


def _call_openai_compatible(
    provider: str,
    prompt: str,
    schema: type[T],
    model: str,
    api_key: str,
    temperature: float = 0.55,
    max_retries: int = 2,
    timeout: int = 120,
    extra_headers: dict | None = None,
) -> T:
    """Shared client for OpenAI-compatible backends (OpenRouter, HuggingFace).

    Both providers accept POST {base_url}/chat/completions with an OpenAI-shaped
    body. We send the full sandwich prompt as a single user message and let the
    model produce the JSON output.
    """
    if provider not in PROVIDERS:
        raise LLMValidationError(f"[llm] unknown provider: {provider}")

    if not api_key:
        raise LLMConnectionError(
            f"[llm] error: {provider} requires an API key. "
            f"Set it in the web UI header or pass --api-key on the CLI."
        )

    base_url = PROVIDERS[provider]["base_url"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    current_prompt = prompt
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            print(f"[llm] {provider} {model}... (attempt {attempt + 1}/{max_retries + 1})")
            start_time = time.time()

            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": current_prompt}],
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=timeout,
            )

            # Surface provider error messages verbatim — they are more useful than
            # a generic "HTTP 400" for the user debugging their token
            if response.status_code == 401:
                raise LLMConnectionError(
                    f"[llm] error: {provider} authentication failed. "
                    f"Check your API key."
                )
            if response.status_code == 429:
                raise LLMConnectionError(
                    f"[llm] error: {provider} rate limit exceeded. "
                    f"Wait a moment or upgrade your plan."
                )
            if response.status_code >= 400:
                try:
                    err_body = response.json()
                except Exception:
                    err_body = response.text
                raise LLMConnectionError(
                    f"[llm] error: {provider} returned {response.status_code}: {err_body}"
                )

            duration = time.time() - start_time
            data = response.json()
            raw_output = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            print(f"[llm] got response in {duration:.1f}s")

            if not raw_output:
                raise LLMValidationError(
                    f"[llm] error: {provider} returned empty response"
                )

            result, error = _parse_and_validate(raw_output, schema)
            if result is not None:
                print("[llm] ✓ response valid")
                return result

            last_error = error
            if attempt < max_retries:
                current_prompt = _build_correction_prompt(raw_output, error, schema)
            else:
                raise LLMValidationError(
                    f"[llm] error: validation failed after {max_retries + 1} attempts. "
                    f"Last error: {last_error}\n\nRaw output:\n{raw_output}"
                )

        except requests.Timeout:
            raise LLMValidationError(
                f"[llm] error: timeout after {timeout}s. Try a smaller model."
            )
        except LLMConnectionError:
            raise  # re-raise as-is
        except requests.RequestException as e:
            raise LLMConnectionError(f"[llm] error: {e}")

    raise LLMValidationError(
        f"[llm] error: unexpected state after {max_retries + 1} attempts"
    )


def call_openrouter(
    prompt: str,
    schema: type[T],
    model: str,
    api_key: str,
    temperature: float = 0.55,
    max_retries: int = 2,
    timeout: int = 120,
) -> T:
    """Call OpenRouter (https://openrouter.ai) and validate against schema.

    OpenRouter proxies dozens of models behind a single OpenAI-compatible API.
    Free tier models have the `:free` suffix. Get a key at
    https://openrouter.ai/keys
    """
    # OpenRouter recommends adding referrer headers for analytics and priority
    extra_headers = {
        "HTTP-Referer": "https://github.com/jrbs/Insights-Extract",
        "X-Title": "Insights Extract",
    }
    return _call_openai_compatible(
        "openrouter", prompt, schema, model, api_key,
        temperature, max_retries, timeout, extra_headers,
    )


def call_huggingface(
    prompt: str,
    schema: type[T],
    model: str,
    api_key: str,
    temperature: float = 0.55,
    max_retries: int = 2,
    timeout: int = 120,
) -> T:
    """Call HuggingFace Inference Providers (OpenAI-compatible) and validate.

    Uses the unified router at https://router.huggingface.co/v1, which routes
    requests to the best available provider (Together, Replicate, Fireworks, etc.)
    for a given model. Get a token at https://huggingface.co/settings/tokens
    """
    return _call_openai_compatible(
        "huggingface", prompt, schema, model, api_key,
        temperature, max_retries, timeout,
    )


def call_llm(
    prompt: str,
    schema: type[T],
    provider: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.55,
    max_retries: int = 2,
    timeout: int | None = None,
) -> T:
    """Dispatcher: route to the right LLM backend based on `provider`.

    This is the main entry point used by the CLI and the web server. Keeping
    the dispatcher thin means every provider has the same retry + validation
    guarantees.

    Args:
        prompt: Sandwich-structured prompt.
        schema: Pydantic model to validate against.
        provider: 'ollama', 'openrouter', or 'huggingface'.
        model: Model name. If None, uses the provider's default.
        api_key: Required for cloud providers. Ignored by ollama.
        temperature: Sampling temperature.
        max_retries: Max retries on invalid JSON.
        timeout: Per-request timeout in seconds.

    Returns:
        Validated schema instance.
    """
    if provider not in PROVIDERS:
        raise LLMValidationError(
            f"[llm] unknown provider '{provider}'. "
            f"Choose one of: {list(PROVIDERS.keys())}"
        )

    if model is None:
        model = PROVIDERS[provider]["default_model"]

    # Local models need a bigger budget than cloud — 5min for Ollama, 2min for cloud
    if timeout is None:
        timeout = 300 if provider == "ollama" else 120

    if provider == "ollama":
        return call_ollama(prompt, schema, model, temperature, max_retries, timeout)
    if provider == "openrouter":
        return call_openrouter(
            prompt, schema, model, api_key or "", temperature, max_retries, timeout
        )
    if provider == "huggingface":
        return call_huggingface(
            prompt, schema, model, api_key or "", temperature, max_retries, timeout
        )

    # Not reachable — guard for future providers
    raise LLMValidationError(f"[llm] provider '{provider}' not implemented")

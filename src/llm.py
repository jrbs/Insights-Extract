"""Ollama HTTP client with schema validation and retry logic.

Comunicação com Ollama local (localhost:11434). Suporta retry automático com
prompts corretivos se a validação falhar.
"""

import json
import time
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class OllamaConnectionError(Exception):
    """Raised when Ollama server is not responding."""

    pass


class OllamaValidationError(Exception):
    """Raised when LLM output fails schema validation after retries."""

    pass


def call_ollama(
    prompt: str,
    schema: type[T],
    model: str = "qwen2.5:7b",
    temperature: float = 0.3,
    max_retries: int = 2,
    timeout: int = 120,
) -> T:
    """Call Ollama and validate response against Pydantic schema.

    Arquitetura sandwich: dados → transcrição → instrução final com schema JSON.
    Se o LLM devolver JSON inválido:
    1. Parse attempt (Tentativa 1: raw JSON)
    2. Retry com prompt corretivo (Tentativa 2: mensagem de erro + schema)
    3. Fail com mensagem clara (max_retries atingido)

    Args:
        prompt: Full prompt with sandwich structure (system + data + instruction)
        schema: Pydantic model class to validate response against
        model: Ollama model name (default: qwen2.5:7b)
        temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
        max_retries: Max retry attempts for invalid output (default: 2)
        timeout: Request timeout in seconds (default: 120)

    Returns:
        Validated instance of schema class

    Raises:
        OllamaConnectionError: If Ollama is not responding
        OllamaValidationError: If output fails validation after all retries
    """
    base_url = "http://localhost:11434"

    # Verificar se Ollama está rodando
    try:
        health = requests.get(f"{base_url}/api/tags", timeout=5)
        health.raise_for_status()
    except requests.RequestException:
        raise OllamaConnectionError(
            "[llm] error: Ollama not responding at localhost:11434. "
            "Start Ollama with: `ollama serve` or `ollama run qwen2.5:7b`"
        )

    # Tentar chamar LLM com retry
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            print(f"[llm] calling {model}... (attempt {attempt + 1}/{max_retries + 1})")
            start_time = time.time()

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()

            duration = time.time() - start_time
            data = response.json()
            raw_output = data.get("response", "").strip()

            print(f"[llm] got response in {duration:.1f}s")

            # Tentar fazer parse do JSON
            try:
                # Extract JSON block if surrounded by markdown
                json_start = raw_output.find("{")
                json_end = raw_output.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_output[json_start:json_end]
                else:
                    json_str = raw_output

                parsed = json.loads(json_str)
                # Validar com Pydantic
                result = schema(**parsed)
                print("[llm] ✓ response valid")
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                if attempt < max_retries:
                    # Construir prompt corretivo (sandwich technique again)
                    correction_prompt = f"""{prompt}

[ERROR IN PREVIOUS RESPONSE]
{str(e)}

[CORRECTION REQUIRED]
You MUST return a valid JSON object that matches this schema exactly:
{schema.model_json_schema()}

Return ONLY the JSON object, no markdown, no additional text."""
                    prompt = correction_prompt
                else:
                    # Max retries exhausted
                    raise OllamaValidationError(
                        f"[llm] error: validation failed after {max_retries + 1} attempts. "
                        f"Last error: {last_error}\n\nRaw output:\n{raw_output}"
                    )

        except requests.Timeout:
            raise OllamaValidationError(
                f"[llm] error: timeout after {timeout}s. "
                f"Try a smaller model or increase timeout. "
                f"Suggestion: ollama pull llama2 (smaller/faster)"
            )
        except requests.RequestException as e:
            raise OllamaConnectionError(f"[llm] error: {e}")

    # Não deve chegar aqui, mas por segurança
    raise OllamaValidationError(
        f"[llm] error: unexpected state after {max_retries + 1} attempts"
    )

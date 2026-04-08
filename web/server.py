"""HTTP server for the Insights Extract web UI.

Serves the single-file HTML frontend and exposes a simple REST API that wraps
the extraction pipeline from src/extract.py.

Uso:
    python -m web.server
    python -m web.server --port 8765

Depois abra: http://localhost:8765
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Import the extract pipeline (parent src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extract import extract as run_extract, validate_input
from src.extract import build_prompt
from src.llm import (
    LLMConnectionError,
    LLMValidationError,
    PROVIDERS,
    call_llm,
)
from src.schema import Insight
from src.transcribe import transcribe


class InsightExtractHandler(BaseHTTPRequestHandler):
    """HTTP handler serving static frontend + /api endpoints."""

    def log_message(self, format, *args):
        """Prefix log output with [web] so it matches pipeline conventions."""
        print(f"[web] {self.address_string()} - {format % args}")

    def _send_json(self, status: int, payload: dict) -> None:
        """Write a JSON response."""
        body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        """Stream a static file (used for index.html)."""
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json(404, {"error": f"not found: {path.name}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        """Route GET requests."""
        if self.path in ("/", "/index.html"):
            self._send_file(Path(__file__).parent / "index.html", "text/html; charset=utf-8")
            return

        if self.path == "/api/health":
            self._send_json(200, {"status": "ok", "timestamp": datetime.utcnow().isoformat()})
            return

        if self.path == "/api/providers":
            # Expose the catalog so the frontend can build its selectors
            # without hardcoding anything. Keeps backend and frontend in sync.
            self._send_json(200, {
                "providers": {
                    name: {
                        "requires_key": cfg["requires_key"],
                        "default_model": cfg["default_model"],
                    }
                    for name, cfg in PROVIDERS.items()
                }
            })
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        """Route POST requests."""
        if self.path != "/api/extract":
            self._send_json(404, {"error": "not found"})
            return

        # Read body
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        input_value = (body.get("input") or "").strip()
        provider = (body.get("provider") or "ollama").strip()
        model = (body.get("model") or "").strip() or None
        api_key = (body.get("api_key") or "").strip() or None

        if not input_value:
            self._send_json(400, {"error": "missing 'input' field"})
            return

        if provider not in PROVIDERS:
            self._send_json(400, {
                "error": f"unknown provider '{provider}'. "
                         f"Choose: {list(PROVIDERS.keys())}"
            })
            return

        if PROVIDERS[provider]["requires_key"] and not api_key:
            self._send_json(400, {
                "error": f"provider '{provider}' requires an API key. "
                         f"Configure it in the UI settings."
            })
            return

        # Run the pipeline end-to-end and return the Insight
        try:
            result = self._run_pipeline(input_value, provider, model, api_key)
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except LLMConnectionError as e:
            self._send_json(503, {"error": str(e)})
        except LLMValidationError as e:
            self._send_json(502, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"internal error: {e}"})

    def _run_pipeline(
        self,
        input_value: str,
        provider: str,
        model: str | None,
        api_key: str | None,
    ) -> dict:
        """Run the extraction pipeline and return the validated Insight dict.

        This duplicates part of src/extract.extract() to avoid calling sys.exit
        from inside the HTTP handler (which would kill the server).
        """
        import time

        # Validate input (CLI may call sys.exit — we catch it and re-raise)
        try:
            input_type, file_path, duration = validate_input(input_value)
        except SystemExit as e:
            raise ValueError(f"invalid input: exit code {e.code}")

        if duration > 180 * 60:
            raise ValueError(f"video too long ({duration // 60} min, max 180)")

        print(f"[web] transcribing ({duration}s)...")
        transcribe_start = time.time()
        transcript = transcribe(file_path, model_name="base")
        transcribe_duration = time.time() - transcribe_start

        if not transcript:
            raise ValueError("no audio detected in file")

        effective_model = model or PROVIDERS[provider]["default_model"]
        print(f"[web] calling LLM [{provider}] {effective_model}...")
        prompt = build_prompt(transcript)
        llm_start = time.time()
        insight = call_llm(
            prompt=prompt,
            schema=Insight,
            provider=provider,
            model=effective_model,
            api_key=api_key,
        )
        llm_duration = time.time() - llm_start

        # Patch metadata with measured values (the LLM may have hallucinated them)
        insight.metadata.transcription_duration_seconds = transcribe_duration
        insight.metadata.llm_duration_seconds = llm_duration
        insight.metadata.llm_model = f"{provider}:{effective_model}"
        insight.metadata.extracted_at = datetime.utcnow()

        return json.loads(insight.model_dump_json())


def main() -> None:
    """Start the HTTP server on the configured port."""
    parser = argparse.ArgumentParser(description="Insights Extract web UI")
    parser.add_argument("--port", "-p", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), InsightExtractHandler)
    print(f"[web] insights-extract UI running at http://{args.host}:{args.port}")
    print(f"[web] open this URL in your browser")
    print(f"[web] press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()

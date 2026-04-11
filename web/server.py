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
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Upload limits — reject files bigger than this to protect the local machine.
# 500MB is enough for most videos that fit the 180min pipeline limit.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"

# Accepted extensions — MUST match src/extract.py::validate_input
ALLOWED_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi",  # video
    ".wav", ".mp3", ".m4a", ".flac",           # audio
}


def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime.

    Thin wrapper around datetime.now(timezone.utc) so the rest of the file
    stays readable. Avoids the Python 3.12 DeprecationWarning from utcnow().
    """
    return datetime.now(timezone.utc)

# Import the extract pipeline (parent src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extract import extract as run_extract, validate_input
from src.prompts import build_prompt
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
            self._send_json(200, {"status": "ok", "timestamp": _utcnow().isoformat()})
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
        if self.path == "/api/upload":
            self._handle_upload()
            return

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

    def _handle_upload(self) -> None:
        """Accept a raw binary file upload and stream it to disk.

        Wire format:
          POST /api/upload
          Content-Type: application/octet-stream
          Content-Length: <bytes>
          X-Filename: <url-encoded original filename>
          <body = raw file bytes>

        Why raw bytes instead of multipart/form-data? The stdlib HTTP server
        does not natively parse multipart, and for a single-file upload the
        raw-body pattern is simpler, faster, and has no extra dependency.
        """
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "empty upload"})
            return

        if length > MAX_UPLOAD_BYTES:
            self._send_json(
                413,
                {"error": f"file too large ({length // (1024 * 1024)} MB, "
                          f"max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"},
            )
            return

        # Extract and sanitize filename
        raw_name = self.headers.get("X-Filename", "upload.bin")
        try:
            original_name = urllib.parse.unquote(raw_name)
        except Exception:
            original_name = "upload.bin"

        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            self._send_json(
                400,
                {"error": f"unsupported file type '{ext}'. "
                          f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"},
            )
            return

        # Generate a collision-free name but keep the extension so that
        # src/extract.py::validate_input recognizes the file type downstream
        safe_name = f"{uuid.uuid4().hex[:12]}{ext}"
        UPLOADS_DIR.mkdir(exist_ok=True)
        target = UPLOADS_DIR / safe_name

        # Stream the body to disk in chunks — avoids loading the whole file
        # into memory for large videos
        try:
            with open(target, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
        except Exception as e:
            # Cleanup partial file on failure
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass
            traceback.print_exc()
            self._send_json(500, {"error": f"write failed: {e}"})
            return

        # Return the absolute path — the frontend will pass it straight to
        # /api/extract as the `input` field. Absolute means no cwd confusion.
        self._send_json(200, {
            "path": str(target.resolve()),
            "size": length,
            "name": original_name,
        })

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

        input_type, file_path, duration, title = validate_input(input_value)

        if duration > 180 * 60:
            raise ValueError(f"video too long ({duration // 60} min, max 180)")

        whisper_model = "small"
        print(f"[web] transcribing with whisper-{whisper_model} ({duration}s)...")
        transcribe_start = time.time()
        transcription = transcribe(file_path, model_name=whisper_model)
        transcribe_duration = time.time() - transcribe_start

        if not transcription.text:
            raise ValueError("no audio detected in file")

        effective_model = model or PROVIDERS[provider]["default_model"]
        print(f"[web] calling LLM [{provider}] {effective_model}...")
        prompt = build_prompt(
            transcription.text,
            source_type=input_type,
            source_url_or_path=input_value,
            source_title=title,
            duration_seconds=duration,
            language_detected=transcription.language,
            segments=transcription.segments,
        )
        llm_start = time.time()
        insight = call_llm(
            prompt=prompt,
            schema=Insight,
            provider=provider,
            model=effective_model,
            api_key=api_key,
        )
        llm_duration = time.time() - llm_start

        # Inject source + metadata com valores reais (nao confiar no LLM)
        insight.source.type = input_type
        insight.source.url_or_path = input_value
        insight.source.duration_seconds = duration
        if title:
            insight.source.title = title

        insight.metadata.transcription_duration_seconds = transcribe_duration
        insight.metadata.llm_duration_seconds = llm_duration
        insight.metadata.llm_model = f"{provider}:{effective_model}"
        insight.metadata.extracted_at = _utcnow()
        insight.metadata.language_detected = transcription.language
        insight.metadata.transcription_model = f"whisper-{whisper_model}"

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

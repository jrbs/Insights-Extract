# Web UI

A small single-file web frontend for `insights-extract`. Useful if you prefer clicking to typing CLI commands, or if you want to show the tool to someone who doesn't live in a terminal.

## What it looks like

- A clean input card (YouTube URL or click-to-pick local file uploader)
- **Provider selector** — Ollama (local, default), OpenRouter, or HuggingFace
- **Per-provider model selector** with curated lists (qwen, llama, claude, gpt-4o, mixtral, etc.)
- **Settings drawer** to paste your OpenRouter / HuggingFace token — saved locally in the browser, never sent to any third party
- Live progress while Whisper transcribes and the LLM extracts
- Result view with color-coded decision card, summary, core thesis, key concepts, caveats, open questions, actionable takeaways, and notable quotes
- Copy / download JSON buttons
- Collapsible raw JSON inspector

Dark theme, no build step, no npm, no bundler. Single HTML file + single Python HTTP handler.

## How to run

> **One process, not two.** `python -m web.server` is both the front-end (serves `index.html`) **and** the back-end (exposes `/api/*` endpoints). Same port, same process. You don't need a second terminal for a separate API server.

From the repo root:

```bash
# 1. Same setup as the CLI — skip if you already did it
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b

# 2. Make sure Ollama is running (in another terminal, or as a background service)
ollama serve &

# 3. Start the web server — serves HTML + /api/* on port 8765
python -m web.server

# 4. Open it in a browser
open http://localhost:8765
```

You can also bind to a different host/port:

```bash
python -m web.server --host 0.0.0.0 --port 9000
```

### Quick smoke test (before opening the browser)

```bash
# Terminal where the server is running should show "[web] insights-extract UI running at http://127.0.0.1:8765"

# In another terminal, hit the health and providers endpoints
curl http://localhost:8765/api/health
curl http://localhost:8765/api/providers
```

If both return JSON, the back-end is wired correctly — just open the URL in the browser.

## How it is wired

```
Browser (web/index.html)
   ↓ POST /api/extract { input, provider, model, api_key }
web/server.py
   ↓ calls src/extract.py + src/llm.py functions directly
   ├─ validate_input()  → checks URL / file
   ├─ transcribe()      → Whisper local (always)
   ├─ call_llm()        → dispatches to ollama / openrouter / huggingface
   │                       with schema validation + retry
   └─ Insight (pydantic) → JSON response
```

The server reuses the exact same pipeline as the CLI. There is no duplicated logic — if the CLI works, the web UI works. The API key you paste in the settings drawer lives in the browser's `localStorage` and is only sent to the backend on the request itself — the server never persists it.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve the HTML frontend |
| GET | `/api/health` | Health check (used by the header indicator) |
| GET | `/api/providers` | List supported providers and default models |
| POST | `/api/extract` | Run the pipeline and return an `Insight` |

### POST /api/extract

Request body:

```json
{
  "input": "https://www.youtube.com/watch?v=XXXXX",
  "provider": "ollama",
  "model": "qwen2.5:7b",
  "api_key": null
}
```

- `provider` — one of `ollama`, `openrouter`, `huggingface`. Default: `ollama`.
- `model` — optional. If omitted, uses the provider's default model.
- `api_key` — required for cloud providers. Ignored by `ollama`. Never persisted server-side.

Response: the full `Insight` schema (see [`SPEC.md`](../SPEC.md)).

Errors are returned with appropriate HTTP status codes:
- `400` — invalid input (malformed URL, file not found, video too long, missing API key)
- `502` — LLM returned invalid JSON after retries
- `503` — LLM backend unreachable (Ollama not running, bad API key, or network failure)

## Design decisions

- **No framework.** Alpine.js + Tailwind via CDN is enough for an interface with one input, one progress view, and one result view. Anything bigger deserves its own repo.
- **No build step.** Copy, open, run. Matches the philosophy of the CLI.
- **No persistence.** The UI is stateless — close the tab and everything is gone. If you want a history, save the JSON.
- **Synchronous backend.** The POST blocks until the pipeline finishes. A 5-minute video locks the UI for 1-2 minutes. For a single-user local tool this is the right trade-off — adding async + job queue would be infrastructure with no user benefit.
- **Progress is cosmetic.** The backend runs synchronously, so the progress steps on the frontend are time-based, not event-based. They exist so the user has something to look at while Whisper runs.

## Why a separate folder

`web/` is deliberately isolated from `src/`. If you only want the CLI, you can ignore this folder. If you want to fork the UI without touching the core, everything is here. The import of `src/` is one-way: `web/` depends on `src/`, never the reverse.

# insights-extract

> **How do you extract structured insights from YouTube videos using a local LLM?** This is the script I wrote to answer that question for myself — and the first episode of a build-in-public series on turning a learning POC into a working QA + Architecture system.

A small Python script that takes a YouTube URL (or a local video file), transcribes it locally with Whisper, and uses a local LLM (Ollama + qwen2.5:7b) to return a structured JSON with the key concepts, architectural risks, open questions, and a clear "is this worth watching in full?" decision.

**No cloud. No API keys. No external services.** Runs end-to-end on a laptop.

---

## Why this exists

Most "AI summarizer" tools give you prose. Prose looks impressive and helps with nothing. When you watch 200+ technical talks a year, what you need is a **decision** — should I spend the next 60 minutes here, or skip — and a **structured artifact** you can search later.

The interesting thing isn't the script itself (it's 80 lines). The interesting thing is the principle behind it: **when you treat an LLM's output as a contract, the LLM becomes a component you can trust. When you leave it open, it becomes a storyteller.**

This repo is the first concrete proof of that principle, published as part of the [Build-in-Public: ArchEngine](#about-the-series) series.

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-handle>/insights-extract.git
cd insights-extract

# 2. Set up the Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Make sure Ollama is running locally and pull the default model
#    (install Ollama from https://ollama.com if you don't have it)
ollama pull qwen2.5:7b

# 4. Verify everything is set up (optional but recommended)
python3 -c "import whisper; import requests; print('[OK] all dependencies installed')"
curl http://localhost:11434/api/tags && echo "[OK] Ollama is running"

# 5. Run it on a YouTube video
python -m src.extract https://www.youtube.com/watch?v=XXXXX

# Or on a local video file
python -m src.extract /path/to/video.mp4

# Save output to file
python -m src.extract <url> --output insights.json
```

The output is a single JSON document printed to stdout. Use `--output insights.json` to save to a file.

### System requirements verified

| Component | Tested version | Status |
|-----------|---|---|
| Python | 3.10, 3.11, 3.12 | ✅ Works |
| yt-dlp | 2024.3.10+ | ✅ Required for YouTube |
| openai-whisper | 20231117+ | ✅ Local transcription |
| pydantic | 2.5.0+ | ✅ Schema validation |
| requests | 2.31.0+ | ✅ Ollama HTTP client |
| ffmpeg | 4.4+ | ✅ Required by Whisper |
| Ollama | 0.1+ | ✅ Local LLM runtime |

If you're on macOS with M-series chip, Whisper will use Metal acceleration automatically (fastest).

## What you get back

A validated JSON object with this shape (full schema in [`SPEC.md`](SPEC.md)):

```json
{
  "schema_version": "1.0.0",
  "decision": {
    "watch_full": true,
    "confidence": "high",
    "rationale": "Short, dense, with real code examples. Worth the 8 minutes."
  },
  "key_concepts": [...],
  "architectural_risks": [...],
  "open_questions": [...],
  "actionable_items": [...],
  "metadata": {...}
}
```

A **structured insight** is the result of forcing an LLM to answer inside a predefined schema instead of free prose. It is the smallest unit that lets you make a decision about a piece of content without consuming the content end-to-end.

## Tech stack

| Component | Version | Purpose |
|---|---|---|
| Python | 3.10+ | type hints, modern syntax |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | ≥ 2024.x | YouTube audio download |
| [openai-whisper](https://github.com/openai/whisper) | latest | local speech-to-text |
| [Ollama](https://ollama.com) | ≥ 0.1 | local LLM runtime |
| [qwen2.5:7b](https://ollama.com/library/qwen2.5) | — | default model (swap with `--model`) |
| [pydantic](https://docs.pydantic.dev) | ≥ 2.0 | output schema validation |

You will also need [`ffmpeg`](https://ffmpeg.org/download.html) installed system-wide (Whisper depends on it).

## Trade-offs

Honest list of what this script does badly, so you can decide if it fits your case:

- **Single video at a time.** No batch, no queue, no database. If you want a searchable knowledge base across many videos, that's [Episode 02](#about-the-series) of this series.
- **Local-only by design.** No fallback to cloud LLMs. If your laptop can't run Ollama + a 7B model, this won't work for you. That's a feature, not a bug — the point is zero external dependency.
- **Whisper is slow on CPU.** A 60-minute video takes 5–8 minutes to transcribe on an M-series Mac. Worth it for privacy and zero cost. If you need real-time, this isn't the tool.
- **Output language follows the input.** No automatic translation. If your video is in English, the JSON comes back in English. If it's in Portuguese, it comes back in Portuguese.
- **No retries beyond schema validation.** If the LLM gives invalid JSON twice in a row, the script fails loudly instead of guessing. This is intentional — silent guesses corrupt downstream pipelines.

## How to run on a different model

```bash
python -m src.extract <input> --model llama3.1:8b
python -m src.extract <input> --model qwen2.5:14b
```

Any Ollama-compatible model works. Larger models give better adherence to the schema at the cost of latency.

## About the series

This repo is **Episode 01** of the **Build-in-Public: ArchEngine** series — a public walkthrough of how a local LLM POC turned into a working system that supports my day job in QA and architecture consulting.

| # | Title | Repo |
|---|---|---|
| 01 | YouTube → structured insight (you are here) | this repo |
| 02 | Multiple insights → searchable project context (RAG) | coming soon |
| 03 | Grounded answers with traceable sources | coming soon |
| 04 | Same retrieval, two cognitive lenses (QA vs Architecture) | coming soon |
| 05 | Refinement companion: AI as preparation, not replacement | coming soon |

Each episode publishes a [LinkedIn post in Portuguese](https://www.linkedin.com/in/lhfsouza/) and a paired public repo in English.

## License

[MIT](LICENSE) — use it, fork it, ship it, learn from it. Attribution is welcome but not required.

## Author

Built by **Luiz Souza** ([LinkedIn](https://www.linkedin.com/in/lhfsouza/)) — QA + Architecture engineer working on the intersection of agile delivery and applied AI. If something here helped you, a star or a comment means a lot.

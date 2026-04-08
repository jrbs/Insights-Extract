"""Main CLI for YouTube Insight Extraction.

Orquestra: download → transcrição → prompt → LLM → validação → JSON.

Uso:
    python -m src.extract https://www.youtube.com/watch?v=XXXXX
    python -m src.extract /path/to/video.mp4 --output insights.json
    python -m src.extract video.mp4 --model llama3.1:8b
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import requests

from .llm import OllamaConnectionError, OllamaValidationError, call_ollama
from .schema import Insight, SourceInfo, Decision, KeyConcept, Metadata
from .transcribe import transcribe


# Exit codes (SPEC.md seção 6)
EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_VIDEO_TOO_LONG = 3
EXIT_TRANSCRIBE_ERROR = 4
EXIT_LLM_TIMEOUT = 5
EXIT_LLM_VALIDATION_ERROR = 6
EXIT_OLLAMA_NOT_RUNNING = 7


def get_video_duration_seconds(file_path: str) -> int:
    """Get video duration using ffprobe (comes with ffmpeg).

    Args:
        file_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If ffprobe fails
    """
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(float(result.stdout.strip()))
    except Exception as e:
        raise RuntimeError(f"[extract] error getting duration: {e}")


def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube using yt-dlp.

    Args:
        url: YouTube URL

    Returns:
        Path to downloaded audio file (.wav)

    Raises:
        RuntimeError: If download fails
    """
    try:
        import subprocess

        output_template = "downloads/%(title)s-%(id)s.%(ext)s"
        Path("downloads").mkdir(exist_ok=True)

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "192",
            "-o",
            output_template,
            url,
        ]

        print(f"[extract] downloading audio from YouTube...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(f"[yt-dlp] error: {result.stderr}")

        # Encontrar arquivo baixado
        for wav_file in Path("downloads").glob("*.wav"):
            return str(wav_file)

        raise RuntimeError("[yt-dlp] error: no audio file found after download")

    except FileNotFoundError:
        raise RuntimeError(
            "[yt-dlp] error: yt-dlp not installed. Install with: pip install yt-dlp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("[yt-dlp] error: download timeout (>10 min)")


def validate_url(url: str) -> bool:
    """Validate YouTube URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid YouTube URL
    """
    youtube_patterns = [
        r"https?://(www\.)?youtube\.com/watch\?v=[-\w]+",
        r"https?://youtu\.be/[-\w]+",
        r"https?://m\.youtube\.com/watch\?v=[-\w]+",
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def validate_input(input_path: str) -> tuple[str, str, int]:
    """Validate and identify input type. Returns type, file_path, duration_seconds.

    Args:
        input_path: YouTube URL or file path

    Returns:
        (type, file_path, duration_seconds) where type is 'youtube', 'local_video', or 'local_audio'

    Raises:
        SystemExit: With appropriate exit code if invalid
    """
    # Check if URL
    if input_path.startswith("http"):
        if not validate_url(input_path):
            error_exit(
                "[extract] error: invalid YouTube URL. "
                "Expected: https://www.youtube.com/watch?v=XXX or https://youtu.be/XXX",
                EXIT_INVALID_INPUT,
            )
        try:
            audio_path = download_youtube_audio(input_path)
            duration = get_video_duration_seconds(audio_path)
            return "youtube", audio_path, duration
        except RuntimeError as e:
            error_exit(str(e), EXIT_INVALID_INPUT)

    # Check if local file
    file_path = Path(input_path)
    if not file_path.exists():
        error_exit(
            f"[extract] error: file not found: {input_path}",
            EXIT_INVALID_INPUT,
        )

    # Determine file type
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    video_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

    if file_path.suffix.lower() in video_extensions:
        input_type = "local_video"
    elif file_path.suffix.lower() in audio_extensions:
        input_type = "local_audio"
    else:
        error_exit(
            f"[extract] error: unsupported file type {file_path.suffix}. "
            f"Supported: {audio_extensions | video_extensions}",
            EXIT_INVALID_INPUT,
        )

    # Get duration
    try:
        duration = get_video_duration_seconds(input_path)
    except RuntimeError as e:
        error_exit(str(e), EXIT_INVALID_INPUT)

    return input_type, input_path, duration


def error_exit(message: str, code: int) -> NoReturn:
    """Print error message and exit with code.

    Args:
        message: Error message
        code: Exit code
    """
    print(message, file=sys.stderr)
    sys.exit(code)


def build_prompt(transcript: str) -> str:
    """Build sandwich-technique prompt with transcript and schema.

    Estrutura sandwich:
    1. System prompt (quem você é, restrições)
    2. Contexto (transcrição)
    3. Instrução final + schema JSON

    Args:
        transcript: Full transcription text

    Returns:
        Complete prompt ready for LLM
    """
    schema_example = Insight.model_json_schema()

    prompt = f"""Você é um assistente técnico especializado em extrair insights estruturados
de conteúdo educacional e técnico. Você responde SEMPRE em português
brasileiro, é técnico e preciso, e usa terminologia correta da área quando
identificada. Você SEMPRE devolve um único bloco JSON válido, sem texto
antes ou depois. Você nunca inventa informação que não está na transcrição.
Quando não tiver evidência suficiente para um campo, deixe a lista vazia
ou marque a confiança como "low".

[TRANSCRIPTION]
{transcript}

[INSTRUCTION]
Analise a transcrição acima e extraia insights estruturados. Devolva um JSON
que valida contra este schema EXATO:

{{
  "schema_version": "1.0.0",
  "source": {{
    "type": "<youtube|local_video|local_audio>",
    "url_or_path": "<input source>",
    "title": "<title or null>",
    "duration_seconds": <int>
  }},
  "decision": {{
    "watch_full": <true|false>,
    "confidence": "<low|medium|high>",
    "rationale": "<max 280 chars — cabe num tweet>"
  }},
  "key_concepts": [
    {{"name": "<max 60 chars>", "explanation": "<max 240 chars>", "timestamp_seconds": <int|null>}},
    ...
  ],
  "architectural_risks": ["<string>", ...],
  "open_questions": ["<string>", ...],
  "actionable_items": ["<string>", ...],
  "metadata": {{
    "extracted_at": "<ISO datetime>",
    "transcription_model": "whisper-base",
    "llm_model": "qwen2.5:7b",
    "transcription_duration_seconds": <float>,
    "llm_duration_seconds": <float>,
    "language_detected": "<iso 639-1 code>"
  }}
}}

CRITICAL RULES:
- key_concepts: MUST have 3-5 items. Less than 3 = trivial content. More than 5 = you're summarizing, not extracting.
- open_questions: ALWAYS at least 1. Good QA always finds something to question.
- architectural_risks: Can be empty if content is not technical. Max 5 items.
- actionable_items: Can be empty if content is not executable. Max 7 items.
- confidence: low = you're not sure. medium = reasonable doubt. high = clear pattern in transcript.
- rationale: MUST fit in 280 characters. If longer, you're over-explaining.

Return ONLY the JSON object. No markdown, no code blocks, no explanation."""

    return prompt


def extract(
    input_path: str,
    output_file: str | None = None,
    model: str = "qwen2.5:7b",
) -> int:
    """Main extraction pipeline.

    Args:
        input_path: YouTube URL or file path
        output_file: Output JSON file path (default: stdout)
        model: Ollama model name

    Returns:
        Exit code
    """
    # 1. Validate input
    print(f"[extract] validating input...")
    try:
        input_type, file_path, duration = validate_input(input_path)
    except SystemExit:
        raise

    print(f"[extract] duration: {duration}s ({duration // 60}m {duration % 60}s)")

    # Check duration limits (SPEC.md)
    if duration > 180 * 60:  # 180 minutes
        error_exit(
            f"[extract] error: video too long ({duration // 60}m). "
            f"Limit is 180 minutes.",
            EXIT_VIDEO_TOO_LONG,
        )
    elif duration > 60 * 60:  # 60 minutes
        print(f"[extract] warning: video is long ({duration // 60}m). Processing anyway.")

    # 2. Transcribe
    print(f"[extract] transcribing...")
    transcribe_start = time.time()
    try:
        transcript = transcribe(file_path, model_name="base")
    except RuntimeError as e:
        error_exit(str(e), EXIT_TRANSCRIBE_ERROR)
    transcribe_duration = time.time() - transcribe_start

    if not transcript:
        error_exit(
            "[extract] error: no audio detected in file",
            EXIT_TRANSCRIBE_ERROR,
        )

    print(f"[extract] transcribed {len(transcript)} chars in {transcribe_duration:.1f}s")

    # 3. Build prompt (sandwich technique)
    prompt = build_prompt(transcript)

    # 4. Call LLM
    print(f"[extract] extracting insights with {model}...")
    llm_start = time.time()
    try:
        insight = call_ollama(prompt, Insight, model=model)
    except OllamaConnectionError as e:
        error_exit(str(e), EXIT_OLLAMA_NOT_RUNNING)
    except OllamaValidationError as e:
        error_exit(str(e), EXIT_LLM_VALIDATION_ERROR)
    llm_duration = time.time() - llm_start

    # 5. Update metadata with real values
    insight.metadata.transcription_duration_seconds = transcribe_duration
    insight.metadata.llm_duration_seconds = llm_duration
    insight.metadata.extracted_at = datetime.utcnow()

    # 6. Output
    output_json = json.loads(insight.model_dump_json(by_alias=False, exclude_none=True))

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        print(f"[extract] ✓ written to {output_file}")
    else:
        print(json.dumps(output_json, indent=2, ensure_ascii=False))

    return EXIT_OK


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract structured insights from YouTube videos or local files"
    )
    parser.add_argument(
        "input", help="YouTube URL (https://www.youtube.com/watch?v=...) or file path"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Ollama model name (default: qwen2.5:7b)",
        default="qwen2.5:7b",
    )

    args = parser.parse_args()

    try:
        exit_code = extract(args.input, args.output, args.model)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[extract] interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[extract] error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

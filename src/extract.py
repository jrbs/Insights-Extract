"""Main CLI for YouTube Insight Extraction.

Orquestra: download → transcrição → prompt → LLM → validação → JSON.

Uso:
    # Local (Ollama, default)
    python -m src.extract https://www.youtube.com/watch?v=XXXXX
    python -m src.extract /path/to/video.mp4 --output insights.json
    python -m src.extract video.mp4 --model llama3.1:8b

    # Cloud providers (API key from env var or --api-key)
    python -m src.extract <url> --provider openrouter --model anthropic/claude-3.5-sonnet
    python -m src.extract <url> --provider huggingface --model meta-llama/Llama-3.1-70B-Instruct

Environment variables read (if --api-key not provided):
    OPENROUTER_API_KEY, HUGGINGFACE_API_KEY
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from .llm import (
    LLMConnectionError,
    LLMValidationError,
    PROVIDERS,
    call_llm,
)
from .prompts import build_prompt
from .schema import Insight
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


def download_youtube_audio(url: str) -> tuple[str, str | None]:
    """Download audio from YouTube using yt-dlp.

    Args:
        url: YouTube URL

    Returns:
        (audio_file_path, video_title_or_None)

    Raises:
        RuntimeError: If download fails
    """
    try:
        import subprocess

        # Diretorio isolado por execucao — evita glob pegar arquivos de runs anteriores
        download_dir = Path("downloads") / uuid.uuid4().hex[:8]
        download_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(download_dir / "%(title)s-%(id)s.%(ext)s")

        # Run yt-dlp as a Python module instead of the `yt-dlp` binary.
        # This way the current interpreter's site-packages is used, so it works
        # even when the venv is active but not exported via PATH.
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "192",
            "-o",
            output_template,
            url,
        ]

        print("[extract] downloading audio from YouTube...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if "No module named" in stderr and "yt_dlp" in stderr:
                raise RuntimeError(
                    "[yt-dlp] error: yt_dlp not installed in the current "
                    "Python environment. Install with: pip install yt-dlp"
                )
            raise RuntimeError(f"[yt-dlp] error: {stderr}")

        # Encontrar arquivo baixado e extrair titulo do nome
        for wav_file in download_dir.glob("*.wav"):
            # Template: %(title)s-%(id)s.%(ext)s → parsear titulo
            filename = wav_file.stem  # "Title Here-VIDEO_ID"
            # Video ID do YouTube: 11 chars no final apos ultimo "-"
            parts = filename.rsplit("-", 1)
            title = parts[0] if len(parts) == 2 else filename
            return str(wav_file), title

        raise RuntimeError("[yt-dlp] error: no audio file found after download")

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


def validate_input(input_path: str) -> tuple[str, str, int, str | None]:
    """Validate and identify input type.

    Args:
        input_path: YouTube URL or file path

    Returns:
        (type, file_path, duration_seconds, title_or_None)

    Raises:
        ValueError: With a human-readable message if the input is invalid.
    """
    if input_path.startswith("http"):
        if not validate_url(input_path):
            raise ValueError(
                "[extract] error: invalid YouTube URL. "
                "Expected: https://www.youtube.com/watch?v=XXX or https://youtu.be/XXX"
            )
        try:
            audio_path, title = download_youtube_audio(input_path)
            duration = get_video_duration_seconds(audio_path)
            return "youtube", audio_path, duration, title
        except RuntimeError as e:
            raise ValueError(str(e)) from e

    file_path = Path(input_path)
    if not file_path.exists():
        raise ValueError(f"[extract] error: file not found: {input_path}")

    audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    video_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

    if file_path.suffix.lower() in video_extensions:
        input_type = "local_video"
    elif file_path.suffix.lower() in audio_extensions:
        input_type = "local_audio"
    else:
        raise ValueError(
            f"[extract] error: unsupported file type {file_path.suffix}. "
            f"Supported: {audio_extensions | video_extensions}"
        )

    try:
        duration = get_video_duration_seconds(input_path)
    except RuntimeError as e:
        raise ValueError(str(e)) from e

    return input_type, input_path, duration, None


def error_exit(message: str, code: int) -> NoReturn:
    """Print error message and exit with code."""
    print(message, file=sys.stderr)
    sys.exit(code)


def extract(
    input_path: str,
    output_file: str | None = None,
    model: str | None = None,
    provider: str = "ollama",
    api_key: str | None = None,
    whisper_model: str = "small",
) -> int:
    """Main extraction pipeline.

    Args:
        input_path: YouTube URL or file path
        output_file: Output JSON file path (default: stdout)
        model: LLM model name (provider-specific). If None, uses provider default.
        provider: 'ollama' (local), 'openrouter', or 'huggingface'
        api_key: API key for cloud providers (required if not ollama)
        whisper_model: Whisper model size (default: small)

    Returns:
        Exit code
    """
    # 1. Validate input
    print("[extract] validating input...")
    try:
        input_type, file_path, duration, title = validate_input(input_path)
    except ValueError as e:
        error_exit(str(e), EXIT_INVALID_INPUT)

    print(f"[extract] duration: {duration}s ({duration // 60}m {duration % 60}s)")

    if duration > 180 * 60:
        error_exit(
            f"[extract] error: video too long ({duration // 60}m). Limit is 180 minutes.",
            EXIT_VIDEO_TOO_LONG,
        )
    elif duration > 60 * 60:
        print(f"[extract] warning: video is long ({duration // 60}m). Processing anyway.")

    # 2. Transcribe — retorna TranscriptionResult com text, segments, language
    print(f"[extract] transcribing with whisper-{whisper_model}...")
    transcribe_start = time.time()
    try:
        transcription = transcribe(file_path, model_name=whisper_model)
    except RuntimeError as e:
        error_exit(str(e), EXIT_TRANSCRIBE_ERROR)
    transcribe_duration = time.time() - transcribe_start

    if not transcription.text:
        error_exit("[extract] error: no audio detected in file", EXIT_TRANSCRIBE_ERROR)

    print(f"[extract] transcribed {len(transcription.text)} chars in {transcribe_duration:.1f}s")

    # 3. Build prompt com metadata + segments (sandwich technique)
    prompt = build_prompt(
        transcription.text,
        source_type=input_type,
        source_url_or_path=input_path,
        source_title=title,
        duration_seconds=duration,
        language_detected=transcription.language,
        segments=transcription.segments,
    )

    # 4. Call LLM
    effective_model = model or PROVIDERS[provider]["default_model"]
    print(f"[extract] extracting insights via {provider} ({effective_model})...")
    llm_start = time.time()
    try:
        insight = call_llm(
            prompt=prompt,
            schema=Insight,
            provider=provider,
            model=effective_model,
            api_key=api_key,
        )
    except LLMConnectionError as e:
        error_exit(str(e), EXIT_OLLAMA_NOT_RUNNING)
    except LLMValidationError as e:
        error_exit(str(e), EXIT_LLM_VALIDATION_ERROR)
    llm_duration = time.time() - llm_start

    # 5. Inject source + metadata com valores reais (nao confiar no LLM para isso)
    insight.source.type = input_type
    insight.source.url_or_path = input_path
    insight.source.duration_seconds = duration
    if title:
        insight.source.title = title

    insight.metadata.transcription_duration_seconds = transcribe_duration
    insight.metadata.llm_duration_seconds = llm_duration
    insight.metadata.llm_model = f"{provider}:{effective_model}"
    insight.metadata.extracted_at = datetime.now(timezone.utc)
    insight.metadata.language_detected = transcription.language
    insight.metadata.transcription_model = f"whisper-{whisper_model}"

    # 6. Output
    output_json = json.loads(insight.model_dump_json(by_alias=False, exclude_none=True))

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        print(f"[extract] written to {output_file}")
    else:
        print(json.dumps(output_json, indent=2, ensure_ascii=False))

    return EXIT_OK


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract structured insights from YouTube videos or local files",
        epilog=(
            "Examples:\n"
            "  # Local (Ollama, default)\n"
            "  python -m src.extract https://www.youtube.com/watch?v=XXXXX\n"
            "\n"
            "  # OpenRouter (key from OPENROUTER_API_KEY env var or --api-key)\n"
            "  python -m src.extract <url> --provider openrouter \\\n"
            "      --model anthropic/claude-3.5-sonnet\n"
            "\n"
            "  # HuggingFace (key from HUGGINGFACE_API_KEY env var or --api-key)\n"
            "  python -m src.extract <url> --provider huggingface \\\n"
            "      --model meta-llama/Llama-3.1-70B-Instruct"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", help="YouTube URL (https://www.youtube.com/watch?v=...) or file path"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "--provider", "-p",
        help="LLM provider (default: ollama)",
        choices=list(PROVIDERS.keys()),
        default="ollama",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name (provider-specific). If omitted, uses the provider's default.",
        default=None,
    )
    parser.add_argument(
        "--api-key", "-k",
        help=(
            "API key for cloud providers. If omitted, reads from env var: "
            "OPENROUTER_API_KEY or HUGGINGFACE_API_KEY."
        ),
        default=None,
    )
    parser.add_argument(
        "--whisper-model",
        help="Whisper model size (default: small). Options: tiny, base, small, medium, large",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
    )

    args = parser.parse_args()

    # Resolve API key from env var when using a cloud provider
    api_key = args.api_key
    if api_key is None and args.provider != "ollama":
        env_var = {
            "openrouter": "OPENROUTER_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }.get(args.provider)
        if env_var:
            api_key = os.environ.get(env_var)

    if args.provider != "ollama" and not api_key:
        print(
            f"[extract] error: provider '{args.provider}' requires an API key. "
            f"Pass --api-key or set the env var.",
            file=sys.stderr,
        )
        sys.exit(EXIT_INVALID_INPUT)

    try:
        exit_code = extract(
            input_path=args.input,
            output_file=args.output,
            model=args.model,
            provider=args.provider,
            api_key=api_key,
            whisper_model=args.whisper_model,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[extract] interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[extract] error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

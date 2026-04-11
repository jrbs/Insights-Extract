"""Whisper-based transcription for audio and video files.

Handles both local files and YouTube downloads. Uses OpenAI's Whisper model.
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import whisper


@dataclass
class TranscriptionResult:
    """Structured output from Whisper transcription.

    Carries text, segment timestamps, and detected language so downstream
    code (prompt builder, metadata injector) can use all of it.
    """

    text: str
    segments: list[dict] = field(default_factory=list)  # {"start": float, "end": float, "text": str}
    language: str = "und"  # ISO 639-1, default "undetermined"


def check_ffmpeg() -> None:
    """Check if ffmpeg is installed and available in PATH.

    Raises:
        RuntimeError: If ffmpeg is not found.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "[ffmpeg] error: ffmpeg not found. Install via 'brew install ffmpeg' (macOS) "
            "or 'apt-get install ffmpeg' (Linux)"
        )


def extract_audio_from_video(video_path: str) -> str:
    """Convert video to WAV audio using ffmpeg.

    Args:
        video_path: Path to video file (.mp4, .mkv, .webm, .mov, .avi)

    Returns:
        Path to extracted audio file (.wav)

    Raises:
        RuntimeError: If ffmpeg conversion fails.
    """
    video_path = Path(video_path)
    audio_path = video_path.with_suffix(".wav")

    # Convert to 16kHz mono WAV — lossless format, best input for Whisper
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ar",
        "16000",  # 16 kHz sample rate
        "-ac",
        "1",  # mono
        "-y",  # overwrite output
        str(audio_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[ffmpeg] error: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("[ffmpeg] error: conversion timeout (>5 min)")

    return str(audio_path)


def transcribe(input_path: str, model_name: str = "base") -> TranscriptionResult:
    """Transcribe audio/video to text using Whisper.

    Returns a TranscriptionResult with text, segment timestamps, and detected
    language. Callers that only need text can use result.text.

    Args:
        input_path: Path to audio or video file
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        TranscriptionResult with text, segments, and language

    Raises:
        RuntimeError: If transcription fails or file is missing.
    """
    check_ffmpeg()

    input_file = Path(input_path)

    if not input_file.exists():
        raise RuntimeError(f"[whisper] error: file not found: {input_path}")

    audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    video_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

    if input_file.suffix.lower() in video_extensions:
        print("[whisper] converting video to audio...")
        audio_path = extract_audio_from_video(str(input_file))
    elif input_file.suffix.lower() in audio_extensions:
        audio_path = str(input_file)
    else:
        raise RuntimeError(
            f"[whisper] error: unsupported format {input_file.suffix}. "
            f"Supported: {audio_extensions | video_extensions}"
        )

    try:
        print(f"[whisper] loading model {model_name}...")
        model = whisper.load_model(model_name)
    except RuntimeError as e:
        raise RuntimeError(f"[whisper] error loading model: {e}")

    try:
        print(f"[whisper] transcribing {Path(audio_path).name}...")
        result = model.transcribe(audio_path)
    except RuntimeError as e:
        raise RuntimeError(f"[whisper] error: {e}")

    # Preservar segments com timestamps para o prompt builder usar
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
    ]
    text = " ".join(seg["text"] for seg in segments)
    language = result.get("language", "und")

    print(f"[whisper] detected language: {language}")

    # Limpar arquivo temporário se foi criado por extract_audio_from_video
    if input_file.suffix.lower() in video_extensions:
        try:
            Path(audio_path).unlink()
        except Exception:
            pass

    return TranscriptionResult(text=text.strip(), segments=segments, language=language)

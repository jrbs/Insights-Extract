"""Whisper-based transcription for audio and video files.

Handles both local files and YouTube downloads. Uses OpenAI's Whisper model.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import whisper


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

    # Convert to 16kHz mono WAV for optimal Whisper quality
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ar",
        "16000",  # 16 kHz sample rate
        "-ac",
        "1",  # mono
        "-q:a",
        "9",  # low bitrate (quality-speed tradeoff)
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


def transcribe(input_path: str, model_name: str = "base") -> str:
    """Transcribe audio/video to text using Whisper.

    Tratamento sandwich: dados primeiro, lógica depois.
    Whisper devolve sempre um arquivo JSON com segments (timestamps).
    Aqui a gente junta o texto e descarta os timestamps (vão no JSON final, se houver).

    Args:
        input_path: Path to audio or video file
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Full transcription text

    Raises:
        RuntimeError: If transcription fails or file is missing.
    """
    check_ffmpeg()

    input_file = Path(input_path)

    # Validar arquivo
    if not input_file.exists():
        raise RuntimeError(f"[whisper] error: file not found: {input_path}")

    # Detectar tipo e converter se necessário
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    video_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

    if input_file.suffix.lower() in video_extensions:
        print(f"[whisper] converting video to audio...")
        audio_path = extract_audio_from_video(str(input_file))
    elif input_file.suffix.lower() in audio_extensions:
        audio_path = str(input_file)
    else:
        raise RuntimeError(
            f"[whisper] error: unsupported format {input_file.suffix}. "
            f"Supported: {audio_extensions | video_extensions}"
        )

    # Load Whisper model (automaticamente faz download se não existir)
    try:
        print(f"[whisper] loading model {model_name}...")
        model = whisper.load_model(model_name)
    except RuntimeError as e:
        raise RuntimeError(f"[whisper] error loading model: {e}")

    # Transcrever
    try:
        print(f"[whisper] transcribing {Path(audio_path).name}...")
        result = model.transcribe(audio_path)
    except RuntimeError as e:
        raise RuntimeError(f"[whisper] error: {e}")

    # Extrair texto (concatenar todos os segments)
    transcript = "".join(segment["text"] for segment in result.get("segments", []))

    # Limpar arquivo temporário se foi criado
    if input_file.suffix.lower() in video_extensions:
        try:
            Path(audio_path).unlink()
        except Exception:
            pass

    return transcript.strip()

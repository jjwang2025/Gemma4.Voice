"""Batch helpers for audio discovery."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".mp4"}


def discover_audio_files(path: Path, recursive: bool = False) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS else []
    if not path.exists():
        return []
    pattern = "**/*" if recursive else "*"
    files = [candidate for candidate in path.glob(pattern) if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    return sorted(files)

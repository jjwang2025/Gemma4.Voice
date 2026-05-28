"""Gemma4.Voice package."""

from .model import TranscriptionMode, TranscriptionResult, Variant, transcribe_audio

__all__ = [
    "TranscriptionMode",
    "TranscriptionResult",
    "Variant",
    "transcribe_audio",
]

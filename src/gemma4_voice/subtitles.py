"""Helpers for timestamped transcript post-processing."""

from __future__ import annotations

import re
from pathlib import Path

TIMESTAMP_LINE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)$"
)


def format_srt_timestamp(timestamp: str) -> str:
    return timestamp.replace(".", ",")


def timestamp_text_to_srt(text: str) -> str:
    blocks: list[str] = []
    index = 1
    for line in text.splitlines():
        match = TIMESTAMP_LINE.match(line.strip())
        if not match:
            continue
        start, end, content = match.groups()
        blocks.append(
            f"{index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{content.strip()}"
        )
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def default_output_paths(audio: Path, variant: str, mode: str, output_dir: Path | None = None) -> tuple[Path, Path | None]:
    target_dir = output_dir or audio.parent
    stem = f"{audio.stem}.{variant}.{mode}"
    text_path = target_dir / f"{stem}.txt"
    srt_path = target_dir / f"{stem}.srt" if mode == "timestamps" else None
    return text_path, srt_path

"""CLI for Gemma4.Voice."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .batch import discover_audio_files
from .logging_utils import configure_logging, get_logger
from .model import DEFAULT_MODEL_ID, TranscriptionMode, Variant, transcribe_audio
from .subtitles import default_output_paths, timestamp_text_to_srt

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma4-voice", description="Gemma 4 ONNX CPU speech transcription")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Transcribe a local audio file")
    transcribe.add_argument("--audio", type=Path, required=True)
    transcribe.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    transcribe.add_argument("--variant", choices=("q4", "q4f16", "quantized", "fp16", "base"), default="q4")
    transcribe.add_argument("--mode", choices=("transcribe", "timestamps"), default="transcribe")
    transcribe.add_argument("--max-new-tokens", type=int, default=128)
    transcribe.add_argument("--output-dir", type=Path, default=None)
    transcribe.add_argument("--output-text", type=Path, default=None)
    transcribe.add_argument("--output-srt", type=Path, default=None)

    batch = subparsers.add_parser("batch", help="Transcribe supported audio files from a file or directory")
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    batch.add_argument("--variant", choices=("q4", "q4f16", "quantized", "fp16", "base"), default="q4")
    batch.add_argument("--mode", choices=("transcribe", "timestamps"), default="transcribe")
    batch.add_argument("--max-new-tokens", type=int, default=128)
    batch.add_argument("--output-dir", type=Path, default=None)
    batch.add_argument("--recursive", action="store_true")
    return parser


def _run_transcribe(args: argparse.Namespace) -> int:
    try:
        result = transcribe_audio(
            args.audio,
            model_id=args.model,
            variant=args.variant,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
        )
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Transcription failed for %s", args.audio)
        print(f"error={exc}", file=sys.stderr)
        return 1

    default_text, default_srt = default_output_paths(args.audio, args.variant, args.mode, args.output_dir)
    output_text = args.output_text or default_text
    output_srt = args.output_srt or default_srt
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_text.write_text(result.text + "\n", encoding="utf-8")
    if args.mode == "timestamps" and output_srt is not None:
        output_srt.parent.mkdir(parents=True, exist_ok=True)
        output_srt.write_text(timestamp_text_to_srt(result.text), encoding="utf-8")

    print("\n---------- Model Output ----------\n")
    print(result.text)
    print("\n---------- Files ----------\n")
    print(f"output_text={output_text}")
    if output_srt is not None:
        print(f"output_srt={output_srt}")
    print("\n---------- Performance ----------\n")
    print(f"infer_seconds={result.infer_seconds:.2f}")
    print(f"audio_seconds={result.audio_seconds:.2f}")
    print("rtf_note=infer_seconds/audio_seconds (no download or load time)")
    print(f"real_time_factor={result.real_time_factor:.2f}")
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    files = discover_audio_files(args.input, recursive=args.recursive)
    if not files:
        LOGGER.error("No supported audio files found in %s", args.input)
        return 1

    failures = 0
    for audio_path in files:
        task_args = argparse.Namespace(
            audio=audio_path,
            model=args.model,
            variant=args.variant,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            output_text=None,
            output_srt=None,
        )
        print(f"\n========== {audio_path.name} ==========")
        failures += 0 if _run_transcribe(task_args) == 0 else 1

    if failures:
        LOGGER.warning("Batch completed with %s failure(s)", failures)
        return 1
    LOGGER.info("Batch completed successfully for %s file(s)", len(files))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    if args.command == "transcribe":
        return _run_transcribe(args)
    if args.command == "batch":
        return _run_batch(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# Gemma4.Voice

CPU-first speech transcription and subtitle generation built on top of `onnx-community/gemma-4-E2B-it-ONNX`.

`Gemma4.Voice` is a focused open-source project for running Gemma 4 speech workflows locally with ONNX Runtime. It is designed for practical local usage rather than generic model experimentation.

## What It Does

- transcribes local audio files with Gemma 4 ONNX
- generates plain text transcripts
- generates timestamped transcript text and `.srt` subtitles
- supports batch processing for files and directories
- reports `RTF` using inference time only

## Why This Project Exists

Running Gemma-family speech workflows on CPU through a generic framework stack can be slow and brittle. This project narrows the scope and provides a cleaner deployment-oriented path:

- ONNX Runtime `CPUExecutionProvider`
- explicit multimodal inference loop
- output naming that makes variant comparisons safe
- lightweight logging and clearer CLI failure handling

## Highlights

- supports `base`, `q4`, `q4f16`, `quantized`, and `fp16` variants
- single-file and batch transcription
- plain transcription and timestamped transcription modes
- automatic `.srt` generation in timestamp mode
- unit-tested subtitle conversion and batch discovery helpers

## Installation

```powershell
cd Gemma4.Voice
python -m pip install -e .
```

## Quick Start

Plain transcription:

```powershell
gemma4-voice transcribe --audio ..\speech_test.wav --variant q4 --mode transcribe --max-new-tokens 64
```

Timestamped transcription and SRT generation:

```powershell
gemma4-voice transcribe --audio ..\speech_test.wav --variant q4 --mode timestamps --max-new-tokens 128
```

Batch transcription:

```powershell
gemma4-voice batch --input ..\ --variant q4 --mode timestamps --recursive --output-dir .\outputs
```

Default outputs for `speech_test.wav` with `--variant q4 --mode timestamps`:

- `speech_test.q4.timestamps.txt`
- `speech_test.q4.timestamps.srt`

## CLI

```text
gemma4-voice transcribe \
  --audio <path> \
  [--variant q4] \
  [--mode transcribe|timestamps] \
  [--max-new-tokens 128] \
  [--output-dir <dir>] \
  [--output-text <file>] \
  [--output-srt <file>]

gemma4-voice batch \
  --input <file-or-dir> \
  [--variant q4] \
  [--mode transcribe|timestamps] \
  [--max-new-tokens 128] \
  [--recursive] \
  [--output-dir <dir>]
```

## Documentation

Model overview:

- English: `docs/onnx-community-gemma-4-E2B-it-ONNX.en.md`
- Chinese: `docs/onnx-community-gemma-4-E2B-it-ONNX.zh-CN.md`

Architecture and inference flow:

- English: `docs/architecture-and-inference.en.md`
- Chinese: `docs/architecture-and-inference.zh-CN.md`

Model footprint and resource breakdown:

- English: `docs/model-footprint-and-resource-breakdown.en.md`
- Chinese: `docs/model-footprint-and-resource-breakdown.zh-CN.md`

## Performance Metric

`RTF` is defined as:

```text
infer_seconds / audio_seconds
```

It excludes model download, model load, and cache warmup time.

## Notes

- default output filenames include both `variant` and `mode`
- `timestamps` mode writes both `.txt` and `.srt`
- timestamp output is model-generated, not forced alignment
- audio length is practically limited by upstream Gemma 4 E2B guidance
- the current implementation targets CPU inference only

## Project Layout

```text
Gemma4.Voice/
  docs/
  src/gemma4_voice/
  tests/
  README.md
  pyproject.toml
  LICENSE
```

## License

MIT for the project code. Model weights and model usage remain subject to the upstream model license.

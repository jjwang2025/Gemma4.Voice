# google/gemma-4-12B-it

## Overview

`google/gemma-4-12B-it` is the instruction-tuned 12B Unified model in the Gemma 4 family. It is published by Google as a public Hugging Face repository, uses the `transformers` stack, and carries an `Apache-2.0` license. At the time of this research, the public metadata shows a recent update timestamp of `2026-06-03`, so it should be treated as one of the newest public Gemma 4 releases.

For `Gemma4.Voice`, this model is interesting for a different reason than `onnx-community/gemma-4-E2B-it-ONNX`. The E2B ONNX package is attractive because it is already deployment-shaped for CPU inference. The 12B model is attractive because it is a more capable and more unified multimodal model that may unlock stronger speech understanding, richer post-transcription reasoning, and better multimodal workflows in the future.

## Official Identity

The public Hugging Face metadata reports:

- model ID: `google/gemma-4-12B-it`
- base model: `google/gemma-4-12B`
- architecture: `Gemma4UnifiedForConditionalGeneration`
- model type: `gemma4_unified`
- pipeline tag: `any-to-any`
- library: `transformers`
- license: `Apache-2.0`

The repository layout is the standard Hugging Face layout rather than an ONNX deployment package. The visible sibling files include:

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `processor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `README.md`

That means the official deployment story is currently centered on `transformers` rather than a CPU-first ONNX package.

## What "Unified" Means

The 12B model is marketed as a **Unified** Gemma 4 model. This is one of the most important architectural differences compared with the smaller E-series models.

In the E2B and E4B route, multimodal processing uses dedicated modality encoders before tokens are handed to the language model. In contrast, the 12B Unified model is described as encoder-free:

- raw image patches are projected into the language model embedding space
- raw audio waveforms are projected into the language model embedding space
- all modalities then flow through a single decoder-only transformer

For downstream users, this matters because the model is not just a "bigger ASR model". It is a more general multimodal reasoning model that also supports audio tasks.

## Core Specifications

The public family README and repository configuration expose the following core numbers:

| Property | Value |
| --- | --- |
| Total parameters | `11.95B` |
| Weight precision in published safetensors metadata | `BF16` |
| Layers | `48` |
| Hidden size | `3840` |
| MLP intermediate size | `15360` |
| Attention heads | `16` |
| KV heads | `8` |
| Head dimension | `256` |
| Sliding window | `1024` tokens |
| Vocabulary size | `262144` |
| Supported modalities | `text`, `image`, `audio` |
| Family-level marketed context length | `256K` tokens |
| Current repo `text_config.max_position_embeddings` | `131072` |

The last two lines should be read carefully. The family-level documentation markets the 12B tier as a `256K` context model, but the currently published repository configuration exposes `max_position_embeddings=131072`. For production planning, it is safer to validate the exact context limit in the deployed runtime stack instead of assuming the headline number is always the effective runtime limit.

## Audio Capability

Audio support is one of the main reasons this model matters to `Gemma4.Voice`.

The public Gemma 4 documentation explicitly states that audio is supported on:

- `E2B`
- `E4B`
- `12B`

The upstream README also includes an audio example built with:

- `AutoProcessor`
- `AutoModelForMultimodalLM`
- a chat prompt that embeds an `audio` content block

This means the official `transformers` path already treats audio as a first-class modality. In practical terms, the model is suitable for:

- speech transcription prompts
- speech-to-text translation prompts
- mixed multimodal workflows where audio is combined with broader reasoning

That said, it is still better understood as a multimodal assistant that can transcribe audio, not as a specialized timestamp-alignment engine.

## Benchmark Positioning

The public README places the 12B model in the middle of the Gemma 4 lineup.

Selected benchmark numbers from the official table:

- `MMLU Pro`: `77.2%`
- `AIME 2026 no tools`: `77.5%`
- `LiveCodeBench v6`: `72.0%`
- `GPQA Diamond`: `78.8%`
- `MMMLU`: `83.4%`
- `CoVoST`: `38.5`
- `FLEURS`: `0.069` (lower is better; the official note excludes Chinese)

These numbers reinforce the idea that 12B is not just an audio feature upgrade over E2B. It is a substantially more capable general multimodal model.

## Storage Footprint

The Hugging Face metadata reports `11,959,730,224` BF16 parameters.

That implies the following rough storage picture:

| Item | Approximate size |
| --- | --- |
| Raw BF16 weight payload | `22.28 GiB` |
| Weight-only INT8 equivalent, if converted | `11.14 GiB` |
| Weight-only INT4 equivalent, if converted | `5.57 GiB` |
| Hugging Face repo reported `usedStorage` | `66.85 GiB` |

Important notes:

- `22.28 GiB` is the practical order of magnitude for the published main weight file.
- The INT8 and INT4 numbers above are rough theoretical conversions, not official release artifacts.
- The repo-level `usedStorage` should not be interpreted as the exact local download size for the visible files, but it is still useful as a signal that the full repository footprint is not small.

## Runtime Memory and KV Cache

Raw weight size is only one part of deployment planning. Runtime memory also includes:

- framework overhead
- activation buffers
- modality preprocessing tensors
- generation buffers
- KV cache growth during decoding

Using the published text configuration (`48` layers, `8` KV heads, `head_dim=256`) and a simple BF16 lower-bound estimate, KV cache growth is approximately:

- `393,216` bytes per cached token
- about `0.375 MiB` per token
- about `384 MiB` for `1024` cached tokens
- about `3.0 GiB` for `8192` cached tokens

This is only a simplified lower-bound estimate. Real runtime behavior is more nuanced because Gemma 4 uses hybrid attention, sliding windows, and implementation-specific buffers. Still, the estimate is useful because it shows that long-context generation can materially increase memory pressure even after the main weights are loaded.

## Local Feasibility On This Machine

The current machine state observed during this research:

- total physical RAM: about `31.51 GiB`
- currently available RAM: about `10.44 GiB`
- free space on drive `D:`: about `95.81 GiB`
- `torch`: `2.11.0+cpu`
- CUDA availability: `False`

This leads to a very clear conclusion.

### What is feasible

- reading the model card and planning support: yes
- downloading the official repository: yes, disk space is sufficient
- integrating 12B as a future backend target in docs and design: yes

### What is not a good idea on this machine right now

- treating the raw Hugging Face BF16 checkpoint as a comfortable CPU-first local runtime target
- expecting good practical latency from CPU-only execution
- using long context or large multimodal inputs without severe memory pressure

A `22.28 GiB` weight file plus framework overhead and generation-time buffers is already too close to the total machine memory budget. With only about `10.44 GiB` free RAM at the time of measurement, loading and running the model locally in a stable and responsive way is not realistic.

In the best case, heavy paging might allow partial experiments. In practice, that would be slow, fragile, and not aligned with the CPU-first design goals of this project.

## Is It Worth Downloading Here?

For this machine, the answer depends on the goal.

### Worth it if

- you want the checkpoint cached locally for future stronger hardware
- you want to inspect processor assets and configs offline
- you expect to test future quantized or alternative runtime formats derived from the same model

### Not worth prioritizing if

- your goal is immediate local CPU transcription
- your goal is a production-friendly CPU backend for `Gemma4.Voice`
- your goal is low-latency ASR on this hardware

For the current machine, downloading the raw official model is a storage decision, not a practical inference decision.

## Ecosystem Status and Project Impact

At the time of this research, a quick Hugging Face API search did not surface an obvious public `onnx-community/gemma-4-12B-it-ONNX` repository.

That matters for `Gemma4.Voice` because the current project is explicitly shaped around:

- `onnxruntime`
- CPU execution
- deployment-friendly multimodal inference loops

The official 12B model does not currently map cleanly onto that project shape. Supporting it in this repository would likely require a separate backend path rather than a small extension of the current E2B ONNX implementation.

## Recommendation for Gemma4.Voice

For now, the most sensible positioning is:

- keep `onnx-community/gemma-4-E2B-it-ONNX` as the practical CPU-first backend
- treat `google/gemma-4-12B-it` as a research and future-integration target
- revisit direct support when one of the following becomes available:
  - a deployment-friendly ONNX export
  - a well-supported low-bit runtime path
  - stronger local hardware, especially a larger GPU or much more system RAM

In other words, `Gemma 4 12B` is strategically important, but it is not yet the right default engine for this repository.

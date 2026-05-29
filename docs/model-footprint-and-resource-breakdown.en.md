# Model Footprint and Resource Breakdown

## Scope

This document explains the approximate parameter scale, storage footprint, and runtime memory characteristics of `onnx-community/gemma-4-E2B-it-ONNX` from the perspective of this project.

The focus is the **audio transcription path** used by `Gemma4.Voice`:

- `audio_encoder`
- `embed_tokens`
- `decoder_model_merged`

The vision encoder is included as a reference, but it is not loaded in the current audio-only CLI flow.

## Measurement Basis

The numbers below come from the local Hugging Face snapshot that was actually used in this workspace:

- model: `onnx-community/gemma-4-E2B-it-ONNX`
- snapshot size on disk: `30.82 GiB`
- inspected snapshot revision: `9f4bef82ea6e296bc69f8a2f5939f73af81b07a6`

Three different concepts are separated on purpose:

1. **Raw parameter count**: counted from ONNX graph initializers in the base export.
2. **Storage footprint**: measured from the actual `.onnx` and `.onnx_data*` files on disk.
3. **Runtime working set**: estimated from loaded weights, activations, and KV cache usage during CPU inference.

These values should not be treated as identical.

## Effective Size vs Raw Tensor Count

Gemma 4 E2B is marketed with an **effective** parameter size, not a literal count of every scalar stored across all deployment tensors.

When the ONNX audio path is inspected directly, the base export exposes about **5.427 billion initializer values** across the three graphs used for speech transcription.

That does **not** contradict the E2B naming. It means:

- the Gemma naming scheme is based on effective model size
- the ONNX deployment package contains multiple subgraphs and auxiliary tensors
- raw tensor counts in deployment artifacts are not the same thing as the model family label

## Block-Level Raw Parameter Count

The following counts are derived from the **base ONNX** graph initializers. They are the best approximation in this repository for discussing how much of the model budget is allocated to each major runtime block.

| Block | Approx. raw initializer values | Approx. scale | Share of audio route |
| --- | ---: | ---: | ---: |
| `audio_encoder` | 294,762,669 | 0.295B | 5.43% |
| `embed_tokens` | 2,751,463,434 | 2.751B | 50.70% |
| `decoder_model_merged` | 2,380,427,125 | 2.380B | 43.87% |
| **Total** | **5,426,653,228** | **5.427B** | **100%** |

### Interpretation

- The **embedding path** is the single largest block in raw storage terms.
- The **decoder** is only slightly smaller and still dominates runtime cost.
- The **audio encoder** is comparatively small in parameter count, but it remains a non-trivial block in both disk and runtime memory terms.

## Storage Footprint by Variant

For this project, the most useful storage question is not the size of the entire repository, but the size of the **audio inference route** that must be present for speech transcription.

### Audio Route Only

This table includes:

- `audio_encoder*`
- `embed_tokens*`
- `decoder_model_merged*`

It excludes tokenizer files and the vision encoder.

| Variant | Audio encoder | Embed tokens | Decoder | Total audio route |
| --- | ---: | ---: | ---: | ---: |
| `base` | 1.10 GiB | 10.25 GiB | 8.87 GiB | **20.22 GiB** |
| `q4` | 0.18 GiB | 1.64 GiB | 1.74 GiB | **3.56 GiB** |
| `quantized` | 0.32 GiB | 2.96 GiB | 2.83 GiB | **6.11 GiB** |

### Vision Encoder Reference

The current CLI does not load the vision path for ASR, but the following sizes matter if a future multimodal workflow enables image input:

| Variant | Vision encoder |
| --- | ---: |
| `base` | 0.63 GiB |
| `q4` | 0.10 GiB |
| `quantized` | 0.18 GiB |

If vision is added back into the runtime, the rough combined storage becomes:

- `base`: about `20.84 GiB`
- `q4`: about `3.66 GiB`
- `quantized`: about `6.30 GiB`

## Storage Reduction Summary

Relative to the base audio path:

- `q4` uses about **17.59%** of the base storage footprint
- `quantized` uses about **30.24%** of the base storage footprint
- `q4` saves about **16.66 GiB** compared with `base` on the audio route alone

This is one reason `q4` is so attractive for CPU-first local deployment.

## Runtime Memory Model

Runtime memory is not just the model file size mapped into memory. For CPU inference, the main contributors are:

1. loaded ONNX weights
2. ONNX Runtime internal buffers and allocator arenas
3. input feature tensors
4. embedding and decoder activations
5. autoregressive **KV cache**

### 1. Weight Residency

As a planning heuristic, the weight residency for the audio path is usually in the same order of magnitude as the selected variant's on-disk audio-route size:

- `base`: plan for a **very large** resident weight set, roughly around the 20 GiB class before extra buffers
- `q4`: plan for a **few-GiB** resident weight set
- `quantized`: plan for something between the two

Exact RSS depends on operating system paging, ONNX Runtime memory arenas, and whether all external data files are touched eagerly.

### 2. Audio Input Tensors

The processor configuration uses:

- 16 kHz audio
- 128-dimensional input features
- about 100 audio frames per second in the prepared feature tensor

For a 30-second clip, raw `input_features` memory is still small compared with model weights, on the order of only a few MiB. This means input audio tensors are **not** the dominant memory bottleneck.

### 3. Decoder KV Cache

The decoder export used here exposes one KV cache pair per layer. Based on the ONNX schema:

- 28 sliding-attention layers use cache width `256`
- 7 full-attention layers use cache width `512`
- both key and value caches are stored as `float32` tensors in the inspected export

That leads to an approximate KV cache cost of:

- **86,016 bytes per generated token**
- about **84.0 KiB per token**

Approximate KV cache growth for batch size `1`:

| Generated tokens | Approx. KV cache |
| ---: | ---: |
| 128 | 10.5 MiB |
| 256 | 21.0 MiB |
| 512 | 42.0 MiB |
| 1024 | 84.0 MiB |
| 2048 | 168.0 MiB |
| 4096 | 336.0 MiB |

This is an important deployment detail: even if the model weights are quantized, the current decoder cache path still grows in a much more conventional floating-point form.

### 4. First-Step Activation Peak

The first decoding step is usually the heaviest step because it includes:

- audio feature extraction
- token embedding lookup
- per-layer inputs preparation
- first decoder pass without a populated cache

Later decoding steps are often lighter per step, but the KV cache keeps growing. In practice, this means peak memory is not always reached at exactly the same moment for all prompts and output lengths.

## Practical CPU Planning Guidance

For local CPU deployment, the following guidance is reasonable:

### `base`

- best treated as a high-memory option
- storage alone is around `20.22 GiB` for the audio route
- suitable only when the machine has substantial RAM headroom

### `q4`

- the best deployment starting point for this project
- storage falls to around `3.56 GiB`
- much easier to keep resident on a CPU-first workstation

### `quantized`

- smaller than `base`, but not as small as `q4`
- should not be assumed faster just because it is quantized

## Why Smaller Storage Does Not Always Mean Faster Inference

In local testing for this project, `q4` was the most attractive CPU variant, while the repository's `quantized` variant was much slower in actual speech transcription even though it occupied less disk space than `base`.

That happens because runtime speed depends on more than file size:

- operator selection
- kernel availability in ONNX Runtime
- cache dtype and memory traffic
- graph structure
- CPU vectorization behavior

So the right deployment question is not only "How small is it?" but also "How does this graph execute on this runtime and hardware?"

## What This Means for `Gemma4.Voice`

For the current project scope, the resource story is clear:

- `embed_tokens` and `decoder_model_merged` dominate the model footprint
- `q4` offers the strongest storage reduction for CPU speech transcription
- runtime memory is affected not only by weight files but also by KV cache growth
- timestamp-heavy prompts can increase runtime working set indirectly by generating more tokens

If the project later adds longer-context workflows, larger batch sizes, or multimodal image+audio prompts, the decoder cache and total resident graph set should be reviewed again.

## Caveats

- parameter counts above are derived from ONNX graph initializers, not from the marketing label of the model family
- storage numbers are based on one inspected snapshot revision and may change if the upstream export is refreshed
- runtime memory is partly estimated from graph schema and deployment behavior, not from a fully instrumented profiler trace for every variant
- this document describes the current CPU-oriented implementation of `Gemma4.Voice`, not every possible Gemma 4 runtime configuration

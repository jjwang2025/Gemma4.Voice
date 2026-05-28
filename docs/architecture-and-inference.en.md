# Architecture and Inference Flow

## Purpose of This Document

This document explains how `Gemma4.Voice` uses `onnx-community/gemma-4-E2B-it-ONNX` for CPU-first speech transcription. It focuses on the runtime architecture, data flow, model graph decomposition, and why the implementation is structured the way it is.

## High-Level Pipeline

At a high level, the transcription path looks like this:

1. accept a local audio file path
2. build a multimodal prompt with audio plus text instruction
3. preprocess the audio with `Gemma4Processor`
4. run the ONNX audio encoder
5. run the ONNX embedding graph
6. inject audio features into the text embedding sequence
7. run the merged decoder autoregressively with KV cache
8. decode generated tokens into final text
9. optionally convert timestamped output into SRT

This is a fairly explicit inference loop. That is intentional. It makes runtime behavior easier to understand and easier to adapt than a hidden end-to-end wrapper.

## Why the Graph Is Split

The ONNX repository is split into separate graphs instead of one monolithic model file. In this project, the relevant components are:

- `audio_encoder*.onnx`
- `embed_tokens*.onnx`
- `decoder_model_merged*.onnx`

Each graph has a different role:

- the audio encoder turns acoustic features into model-space audio features
- the embedding graph produces text token embeddings and per-layer inputs
- the decoder graph performs autoregressive text generation

This split has several benefits:

- lower conceptual complexity per component
- easier debugging when a single stage fails
- more flexibility for future optimizations
- the ability to skip unused modalities such as vision in audio-only workflows

## Processor Responsibilities

`Gemma4Processor` is central to the design. It does much more than tokenization.

It is responsible for:

- applying the multimodal chat template
- extracting audio features from the input waveform
- creating `input_ids`, `attention_mask`, and audio feature tensors
- preserving the placement of special audio tokens expected by the model

This means the processor acts as the boundary between user-facing inputs and the lower-level ONNX graphs.

## Audio Feature Path

For audio transcription, the processor creates acoustic feature tensors such as:

- `input_features`
- `input_features_mask`

These are sent to the ONNX audio encoder. The output of that stage is `audio_features`, which are later inserted into the token embedding stream in positions corresponding to the model's audio token.

This is a key architectural detail. The decoder does not consume raw audio directly. Instead, it consumes a text-like sequence whose embedding stream has been augmented with audio-derived features.

## Embedding Stage

The embedding graph produces two important outputs:

- `inputs_embeds`
- `per_layer_inputs`

`inputs_embeds` contains the token embeddings for the current generation step. `per_layer_inputs` carries additional model-specific data needed by the decoder layers.

Once the audio encoder has produced `audio_features`, those features are inserted into `inputs_embeds` at positions matching the audio token. This effectively fuses audio context into the sequence that the decoder will process.

## Decoder Stage and KV Cache

The merged decoder graph performs autoregressive generation. It accepts:

- token embeddings
- attention mask
- position ids
- per-layer inputs
- past key/value tensors

During generation, the decoder produces:

- logits for the next token
- updated present key/value tensors

Those updated tensors are fed back into the next loop iteration as `past_key_values`. This is standard KV-cache behavior and is essential for efficient autoregressive decoding.

Without cache reuse, each generation step would become much more expensive, especially as the sequence grows.

## Why the Project Uses an Explicit Loop

It would be possible to hide all of this behind a helper abstraction, but the current project keeps the loop visible for engineering reasons:

- easier to inspect performance
- easier to debug tensor shape mismatches
- easier to adapt output stopping rules
- easier to experiment with prompt strategies
- easier to support future batch or streaming features

For a deployment-oriented open source project, that explicitness is often more useful than a shorter but more opaque implementation.

## Prompt Modes

The project currently exposes two main prompt styles:

- `transcribe`
- `timestamps`

The model behavior changes because the instruction changes, not because the decoding stack changes.

In other words:

- the same ONNX pipeline is used in both modes
- the difference is the natural-language task instruction

This is important because it shows how Gemma 4 is being used here: as an instruction-following multimodal model, not as a narrow speech decoder with hardcoded output behavior.

## Timestamp Generation vs Alignment

When the project produces timestamped transcription, the timestamps are generated by the model in response to a formatting prompt. They are not derived from a dedicated alignment stage.

That means the architecture supports:

- subtitle-like output
- useful segment-level estimates

But it does not provide:

- deterministic forced alignment
- word-level timing guarantees

This distinction should guide downstream product decisions.

## CPU-First Design Choices

`Gemma4.Voice` is intentionally CPU-first. That affects several design decisions:

- ONNX Runtime is preferred over a generic PyTorch CPU path
- the `q4` variant is treated as a practical default for local testing
- RTF is measured using inference time only
- outputs are written to files immediately for simple offline workflows

This keeps the project focused on a realistic local deployment target instead of turning it into a general-purpose training or fine-tuning codebase.

## Error Handling Strategy

The project uses lightweight but explicit error handling.

The CLI catches:

- missing input files
- runtime exceptions during transcription
- empty batch discovery results

The goal is not to hide failures but to fail clearly and surface enough context to debug the issue.

## Batch Mode Design

Batch mode is intentionally implemented as repeated single-file transcription instead of trying to batch multiple audios into one ONNX inference call.

That is a deliberate tradeoff:

- simpler control flow
- simpler output naming
- easier failure isolation per file
- less risk of introducing modality-shape bugs

For an open source project at this maturity level, that is the better engineering tradeoff than premature batching complexity.

## Output Design

The output strategy is also intentionally simple.

For every input audio file, the project writes:

- a plain `.txt` transcript
- and, in timestamp mode, a `.srt` subtitle file

Default filenames include both the selected variant and mode. This prevents accidental overwrites when users compare model configurations.

## Why This Structure Works Well

This architecture works well for a small deployment-oriented open source project because it balances:

- clarity
- practical local performance
- easy extension
- minimal hidden behavior

It is not the shortest implementation possible, but it is structured for maintainability and debugging, which is usually the better tradeoff for a real project.

"""Gemma 4 ONNX transcription pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, GenerationConfig

from .logging_utils import get_logger
from .prompts import prompt_for_mode

Variant = Literal["q4", "q4f16", "quantized", "fp16", "base"]
TranscriptionMode = Literal["transcribe", "timestamps"]
DEFAULT_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX"
LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    infer_seconds: float
    audio_seconds: float
    real_time_factor: float
    model_id: str
    variant: Variant
    mode: TranscriptionMode
    audio_path: Path


def _select_paths(variant: Variant) -> tuple[str, str, str]:
    suffix = {
        "q4": "_q4",
        "q4f16": "_q4f16",
        "quantized": "_quantized",
        "fp16": "_fp16",
        "base": "",
    }[variant]
    return (
        f"onnx/audio_encoder{suffix}.onnx",
        f"onnx/embed_tokens{suffix}.onnx",
        f"onnx/decoder_model_merged{suffix}.onnx",
    )


def _to_numpy(value: np.ndarray | object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def transcribe_audio(
    audio_path: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    variant: Variant = "q4",
    mode: TranscriptionMode = "transcribe",
    max_new_tokens: int = 128,
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    LOGGER.info("Preparing transcription for %s (variant=%s, mode=%s)", audio_path, variant, mode)

    audio_model, embed_model, decoder_model = _select_paths(variant)

    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)

    model_dir = snapshot_download(
        model_id,
        allow_patterns=[
            f"{audio_model}*",
            f"{embed_model}*",
            f"{decoder_model}*",
        ],
    )
    LOGGER.debug("Using model snapshot at %s", model_dir)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    audio_session = ort.InferenceSession(os.path.join(model_dir, audio_model), sess_options=session_options, providers=providers)
    embed_session = ort.InferenceSession(os.path.join(model_dir, embed_model), sess_options=session_options, providers=providers)
    decoder_session = ort.InferenceSession(os.path.join(model_dir, decoder_model), sess_options=session_options, providers=providers)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt_for_mode(mode)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = _to_numpy(inputs["input_ids"]).astype(np.int64)
    attention_mask = _to_numpy(inputs["attention_mask"]).astype(np.int64)
    position_ids = np.cumsum(attention_mask, axis=-1, dtype=np.int64) - 1
    input_features = _to_numpy(inputs["input_features"]).astype(np.float32)
    input_features_mask = _to_numpy(inputs["input_features_mask"]).astype(np.bool_)

    batch_size = input_ids.shape[0]
    num_logits_to_keep = np.array(1, dtype=np.int64)
    eos_token_id = np.array(generation_config.eos_token_id, dtype=np.int64).reshape(-1)
    audio_token_id = np.int64(config.audio_token_id)
    audio_seconds = input_features.shape[1] * 0.01

    past_key_values: dict[str, np.ndarray] = {}
    for inp in decoder_session.get_inputs():
        if inp.name.startswith("past_key_values"):
            dtype = np.float32 if inp.type == "tensor(float)" else np.float16
            past_key_values[inp.name] = np.zeros([batch_size, inp.shape[1], 0, inp.shape[3]], dtype=dtype)

    audio_features = None
    generated_tokens = np.empty((batch_size, 0), dtype=np.int64)

    infer_started = time.perf_counter()
    for _ in range(max_new_tokens):
        embed_outputs = embed_session.run(None, {"input_ids": input_ids})
        inputs_embeds = embed_outputs[0]
        per_layer_inputs = embed_outputs[1]

        if audio_features is None:
            audio_features = audio_session.run(
                ["audio_features"],
                {"input_features": input_features, "input_features_mask": input_features_mask},
            )[0]
            mask = (input_ids == audio_token_id).reshape(-1)
            flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            flat_embeds[mask] = audio_features
            inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

        outputs = decoder_session.run(
            None,
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "per_layer_inputs": per_layer_inputs,
                "position_ids": position_ids,
                "num_logits_to_keep": num_logits_to_keep,
                **past_key_values,
            },
        )

        logits = outputs[0]
        next_token = logits[:, -1].argmax(-1, keepdims=True).astype(np.int64)
        generated_tokens = np.concatenate([generated_tokens, next_token], axis=-1)
        input_ids = next_token
        attention_mask = np.concatenate([attention_mask, np.ones_like(next_token)], axis=-1)
        position_ids = position_ids[:, -1:] + 1
        for index, key in enumerate(past_key_values):
            past_key_values[key] = outputs[index + 1]

        if np.isin(next_token, eos_token_id).any():
            break

    infer_seconds = time.perf_counter() - infer_started
    decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    LOGGER.info("Completed transcription for %s in %.2fs", audio_path.name, infer_seconds)
    return TranscriptionResult(
        text=decoded,
        infer_seconds=infer_seconds,
        audio_seconds=audio_seconds,
        real_time_factor=infer_seconds / max(audio_seconds, 1e-6),
        model_id=model_id,
        variant=variant,
        mode=mode,
        audio_path=audio_path,
    )

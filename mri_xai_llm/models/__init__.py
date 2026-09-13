"""Model loading/lifecycle management for the MRI XAI pipeline.

Exposes ``ModelManager`` as the single entry point notebooks / the Gradio
app should use to obtain vision, language, and multimodal models. Loading is
lazy and cached per-instance so a model is only downloaded/instantiated the
first time it's actually needed; ``unload_model`` frees GPU memory when a
model is no longer needed (e.g. swapping the LLM for a lighter fallback).

Memory note: on a single Colab GPU, only load what you currently need
(e.g. unload the language model before loading a different one). Vision
encoder + projector + finding head together are small; the LLM is the
memory-dominant piece, hence 4-bit quantization support below.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import torch
import torch.nn as nn

from mri_xai_llm.config.config import MODEL_CONFIG, detect_device
from mri_xai_llm.models.multimodal_model import MRIMultimodalModel
from mri_xai_llm.models.vision_encoder import MRIVisionEncoder

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    BitsAndBytesConfig = None  # type: ignore[assignment,misc]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:
    import bitsandbytes  # noqa: F401
    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    _BITSANDBYTES_AVAILABLE = False


class ModelManager:
    """Lazily loads and caches vision, language, and multimodal models.

    Usage::

        mm = ModelManager()
        vision = mm.load_vision_model()
        lm, tokenizer = mm.load_language_model()
        mm.unload_model(lm)  # free GPU memory when done with it
    """

    def __init__(self) -> None:
        self.device = detect_device()
        self._vision_model: MRIVisionEncoder | None = None
        self._language_model: Any = None
        self._language_tokenizer: Any = None
        self._multimodal_model: MRIMultimodalModel | None = None

    def load_vision_model(self) -> MRIVisionEncoder:
        if self._vision_model is not None:
            return self._vision_model
        logger.info("Loading vision encoder...")
        self._vision_model = MRIVisionEncoder(freeze=True)
        return self._vision_model

    def load_language_model(self) -> tuple[Any, Any]:
        """Loads the causal LM (+ tokenizer), 4-bit quantized on CUDA when configured.

        Falls back, in order: 4-bit quantized primary model -> fp16/fp32
        primary model -> fp16/fp32 fallback model (MODEL_CONFIG
        ["language_model_fallback"]). Any network or dependency failure is
        caught and logged rather than raised, so the rest of the pipeline
        (vision, segmentation) can still run without an LLM available.
        """
        if self._language_model is not None and self._language_tokenizer is not None:
            return self._language_model, self._language_tokenizer

        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is required for load_language_model. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR}"
            )

        primary_name = MODEL_CONFIG["language_model"]
        fallback_name = MODEL_CONFIG["language_model_fallback"]
        use_4bit = MODEL_CONFIG["use_4bit"] and torch.cuda.is_available() and _BITSANDBYTES_AVAILABLE

        model, tokenizer, loaded_name = self._try_load_causal_lm(primary_name, use_4bit)
        if model is None:
            logger.warning("Primary language model '%s' failed to load; trying fallback '%s'.", primary_name, fallback_name)
            model, tokenizer, loaded_name = self._try_load_causal_lm(fallback_name, use_4bit=False)

        if model is None:
            raise RuntimeError(
                f"Could not load either the primary language model '{primary_name}' "
                f"or the fallback '{fallback_name}'. Check network access / model availability."
            )

        logger.info("Language model loaded: %s", loaded_name)
        self._language_model = model
        self._language_tokenizer = tokenizer
        return self._language_model, self._language_tokenizer

    def _try_load_causal_lm(self, model_name: str, use_4bit: bool) -> tuple[Any, Any, str | None]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs: dict[str, Any] = {}
            if use_4bit and BitsAndBytesConfig is not None:
                logger.info("Loading '%s' with 4-bit quantization (bitsandbytes).", model_name)
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = "auto"
            elif torch.cuda.is_available():
                logger.info("Loading '%s' in fp16 on CUDA (no 4-bit quantization).", model_name)
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            else:
                logger.info("Loading '%s' in fp32 on CPU.", model_name)
                load_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if "device_map" not in load_kwargs:
                model = model.to(self.device)
            model.eval()
            return model, tokenizer, model_name
        except Exception as exc:  # noqa: BLE001 - network/weights/quantization failures all fall back
            logger.warning("Failed to load language model '%s': %s", model_name, exc)
            return None, None, None

    def load_multimodal_model(self, llm_hidden_dim: int | None = None) -> MRIMultimodalModel:
        """Builds the composed vision -> projector -> finding-head model.

        If ``llm_hidden_dim`` is not given, the language model is loaded
        first (if not already cached) to read its hidden size from config,
        so the projector output dimension matches exactly.
        """
        if self._multimodal_model is not None:
            return self._multimodal_model

        if llm_hidden_dim is None:
            lm, _ = self.load_language_model()
            llm_hidden_dim = int(getattr(lm.config, "hidden_size", MODEL_CONFIG["projection_hidden_dim"]))

        logger.info("Building multimodal model (llm_hidden_dim=%d)...", llm_hidden_dim)
        self._multimodal_model = MRIMultimodalModel(llm_hidden_dim=llm_hidden_dim)
        self._multimodal_model.to(self.device)
        # Reuse the already-loaded vision encoder if present, to avoid loading twice.
        if self._vision_model is not None:
            self._multimodal_model.vision_encoder = self._vision_model
        else:
            self._vision_model = self._multimodal_model.vision_encoder
        return self._multimodal_model

    def unload_model(self, model: Any) -> None:
        """Frees a model from GPU/CPU memory and clears this manager's cache slot for it.

        Use `torch.autocast(device_type="cuda", dtype=torch.float16)` around
        forward passes for mixed-precision inference/training instead of
        keeping everything in fp32/fp16 statically -- this is handled by
        callers (e.g. training loops), not by ModelManager itself.
        """
        if model is self._vision_model:
            self._vision_model = None
        if model is self._language_model:
            self._language_model = None
            self._language_tokenizer = None
        if model is self._multimodal_model:
            self._multimodal_model = None

        try:
            model.to("cpu")
        except Exception:  # noqa: BLE001 - best-effort; device_map="auto" models may refuse .to()
            pass

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU cache cleared.")

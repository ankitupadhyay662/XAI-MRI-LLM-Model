"""Vision -> structured-findings -> projection composition.

Architecture note: the language model NEVER sees raw pixels. The vision
encoder produces embeddings, the FindingClassifierHead scores them against
a fixed clinical taxonomy *before* any language generation happens, and only
the projected embedding (plus the structured findings, assembled elsewhere
into a text prompt) reach the LLM. This keeps generation grounded in
evidence the system can audit, rather than an opaque image-to-text leap.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from config.config import FINDING_TAXONOMY, MODEL_CONFIG
from models.vision_encoder import MRIVisionEncoder

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - environment guard
    LoraConfig = None  # type: ignore[assignment,misc]
    get_peft_model = None  # type: ignore[assignment,misc]


class MultimodalProjector(nn.Module):
    """Projects the pooled vision embedding into the language model's hidden size.

    This is the primary trainable component alongside LoRA adapters: the
    vision encoder stays frozen (see MRIVisionEncoder), so all visual-to-
    textual grounding adaptation happens here.
    """

    def __init__(self, vision_dim: int, llm_hidden_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or MODEL_CONFIG["projection_hidden_dim"]
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class FindingClassifierHead(nn.Module):
    """Multi-label classifier over FINDING_TAXONOMY, scored from pooled vision features.

    Runs as a distinct, auditable step BEFORE the LLM is invoked: its output
    (a probability per finding) becomes structured evidence that a prompt
    builder turns into text, rather than letting the LLM freely hallucinate
    a diagnosis directly from the image.
    """

    def __init__(self, vision_dim: int, num_classes: int | None = None) -> None:
        super().__init__()
        self.taxonomy = list(FINDING_TAXONOMY)
        num_classes = num_classes or len(self.taxonomy)
        self.classifier = nn.Linear(vision_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        return {"finding_logits": logits, "finding_probs": probs}


class MRIMultimodalModel(nn.Module):
    """Composes vision encoder + projector + finding classifier.

    The language model itself is loaded/managed separately (see
    ``mri_xai_llm.models.ModelManager.load_language_model``) since it is
    often swapped, quantized, or unloaded independently of the vision stack.
    """

    def __init__(self, llm_hidden_dim: int, freeze_vision: bool = True) -> None:
        super().__init__()
        self.vision_encoder = MRIVisionEncoder(freeze=freeze_vision)
        self.projector = MultimodalProjector(self.vision_encoder.hidden_dim, llm_hidden_dim)
        self.finding_head = FindingClassifierHead(self.vision_encoder.hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        vision_out = self.vision_encoder(pixel_values)
        pooled, spatial = vision_out["pooled"], vision_out["spatial"]
        projected = self.projector(pooled)
        finding_out = self.finding_head(pooled)
        return {
            "pooled": pooled,
            "spatial": spatial,
            "projected": projected,
            "finding_logits": finding_out["finding_logits"],
            "finding_probs": finding_out["finding_probs"],
        }

    def apply_lora(self, language_model: nn.Module) -> nn.Module:
        """Wraps ``language_model`` with LoRA adapters per MODEL_CONFIG.

        Target module names vary across LLM architectures (Qwen2, LLaMA,
        T5, ...), so this is best-effort: on failure the original model is
        returned unmodified and a warning is logged rather than crashing the
        pipeline (full fine-tuning / inference-only can still proceed).
        """
        if LoraConfig is None or get_peft_model is None:
            logger.warning("peft is not installed; returning the language model without LoRA adapters.")
            return language_model

        candidate_target_modules = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA / Qwen2 style
            ["query", "key", "value"],  # BERT-style
            ["q", "k", "v", "o"],  # T5 style
        ]

        for targets in candidate_target_modules:
            try:
                lora_config = LoraConfig(
                    r=MODEL_CONFIG["lora_r"],
                    lora_alpha=MODEL_CONFIG["lora_alpha"],
                    lora_dropout=MODEL_CONFIG["lora_dropout"],
                    target_modules=targets,
                    task_type="CAUSAL_LM",
                    bias="none",
                )
                peft_model = get_peft_model(language_model, lora_config)
                logger.info("Applied LoRA adapters with target_modules=%s", targets)
                return peft_model
            except Exception as exc:  # noqa: BLE001 - target module names are architecture-specific
                logger.debug("LoRA target_modules=%s failed: %s", targets, exc)
                continue

        logger.warning(
            "Could not determine LoRA target modules for this language model architecture; "
            "returning the base model unmodified."
        )
        return language_model

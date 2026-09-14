"""Central configuration for the Explainable MRI AI Research System.

Everything downstream (preprocessing, model loading, XAI, reporting) reads
its defaults from here. Values can be overridden at runtime by loading
config.yaml (see load_yaml_overrides) so the notebook / Gradio app never
need hard-coded local paths.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # torch may not be installed yet when config is first imported
    torch = None

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def detect_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def project_root() -> Path:
    """Auto-detect the project root: Colab -> /content, else this file's grandparent."""
    if is_colab():
        root = Path("/content/mri_xai_llm")
    else:
        root = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    return root


ROOT = project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = ROOT / "checkpoints"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model configuration (modular / swappable backends)
# ---------------------------------------------------------------------------
MODEL_CONFIG: dict[str, Any] = {
    # Vision encoder: any timm / torchvision / HF backbone that yields a
    # spatial feature map + pooled embedding. Swap freely.
    "vision_encoder": "microsoft/rad-dino",       # medical-imaging ViT (fallback below)
    "vision_encoder_fallback": "vit_base_patch16_224",  # torchvision/timm fallback if HF weights unavailable
    "segmentation_model": "monai_unet",           # MONAI UNet, randomly-initialized research demo
    "language_model": "Qwen/Qwen2.5-1.5B-Instruct",  # lightweight instruction LLM, Colab-friendly
    "language_model_fallback": "google/flan-t5-base",
    "projection_hidden_dim": 1024,
    "image_size": 224,
    "max_new_tokens": 1024,
    "use_4bit": True,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}

# ---------------------------------------------------------------------------
# Hallucination-control thresholds (NOT clinically validated)
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.60  # findings below this are relabeled "uncertain, requires expert review"

# ---------------------------------------------------------------------------
# Finding taxonomy (configurable)
# ---------------------------------------------------------------------------
FINDING_TAXONOMY = [
    "Normal",
    "Abnormal",
    "Mass/Lesion",
    "Edema",
    "Hemorrhage",
    "Infarction",
    "White matter abnormality",
    "Atrophy",
    "Hydrocephalus",
    "Midline shift",
    "Ventricular abnormality",
    "Restricted diffusion",
    "Enhancement",
    "Fracture",
    "Degenerative change",
    "Other",
    "Unknown",
]

ANATOMICAL_REGIONS = [
    "brain", "frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe",
    "cerebellum", "brainstem", "ventricles", "skull", "spine", "spinal cord",
    "discs", "joints", "other",
]

SUPPORTED_SEQUENCES = [
    "T1", "T1CE", "T2", "FLAIR", "DWI", "ADC", "SWI", "GRE", "PD", "STIR",
    "MRA", "MRV", "Unknown",
]

DISCLAIMER = (
    "Research/educational output — not a medical diagnosis. Results must be "
    "reviewed and validated by a qualified radiologist/physician."
)


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    seed: int = SEED
    min_confidence: float = MIN_CONFIDENCE
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 1
    extra: dict = field(default_factory=dict)


def load_yaml_overrides(path: str | Path) -> dict[str, Any]:
    """Load config.yaml and merge into MODEL_CONFIG / thresholds at runtime."""
    import yaml
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    if "model" in overrides:
        MODEL_CONFIG.update(overrides["model"])
    global MIN_CONFIDENCE
    if "min_confidence" in overrides:
        MIN_CONFIDENCE = float(overrides["min_confidence"])
    return overrides

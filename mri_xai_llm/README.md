# Explainable AI MRI Analysis and Medical Report Generation System

> **Research/educational output — not a medical diagnosis. Results must be
> reviewed and validated by a qualified radiologist/physician.**

## 1. Objective

A research-grade, explainable pipeline that takes an MRI image/series
(DICOM, NIfTI, or standard image formats), analyzes it with a medical
vision model, detects imaging findings against a configurable taxonomy,
explains those findings with multiple XAI methods (Grad-CAM, Integrated
Gradients, attention rollout, occlusion sensitivity), and generates a
structured, radiology-style report — grounded in the imaging evidence, not
invented by an LLM. This is a prototype for research and education. It is
**not** a clinical diagnostic device and must never be presented as one.

## 2. Architecture

```
MRI Images / DICOM / NIfTI
          |
   Data Validation
          |
   MRI Preprocessing (orientation, resampling, intensity normalization,
                       skull/body region handling, slice selection)
          |
   MRI Vision Encoder --> Feature Extraction
          |                     |
          v                     v
   Finding Detection      Segmentation
          |                     |
          +----------+----------+
                     |
                 XAI Layer (Grad-CAM, Integrated Gradients,
                            Attention Rollout, Occlusion)
                     |
       Structured Evidence (Pydantic-validated findings + XAI regions)
                     |
                    LLM (grounded report generation, deterministic fallback)
                     |
        Structured Medical Report + Confidence/Uncertainty
                     |
                Gradio Web UI (Colab)
```

The system never sends a raw MRI slice straight into a text LLM and asks
"what disease is this?". The LLM only ever sees structured, confidence-
filtered, evidence-verified findings — see [`reporting/report_generator.py`](mri_xai_llm/reporting/report_generator.py).

### Repository layout

```
mri_xai_llm/
├── config/            config.py, runtime defaults + config.yaml overrides
├── data/               raw/ processed/ sample/
├── models/             vision_encoder.py, segmentation.py, multimodal_model.py, ModelManager
├── preprocessing/       dicom.py, nifti.py, transforms.py
├── xai/                gradcam.py, integrated_gradients.py, attention.py, occlusion.py
├── reporting/           schemas.py (Pydantic), findings.py, report_generator.py
├── evaluation/          classification.py, segmentation.py, report_metrics.py
├── ui/                  gradio_app.py
└── main.py
```

## 3. Installation

```bash
pip install -r requirements.txt
```

Designed to run top-to-bottom in **Google Colab** with a GPU runtime
(`Runtime > Change runtime type > GPU`). The notebook [`MRI_XAI_LLM.ipynb`](MRI_XAI_LLM.ipynb)
auto-detects Colab vs. local paths and CPU vs. CUDA — no hard-coded paths.

## 4. Configuration

All model/preprocessing/XAI/threshold settings live in
[`config.yaml`](config.yaml) and [`mri_xai_llm/config/config.py`](mri_xai_llm/config/config.py).
Swap the vision encoder, language model, or thresholds without touching
source code:

```yaml
model:
  vision_encoder: "microsoft/rad-dino"
  language_model: "Qwen/Qwen2.5-1.5B-Instruct"
min_confidence: 0.60
```

## 5. Dataset preparation

The system does not ship with patient data. Datasets are described and
adapted, not bundled:

| Dataset | Contents | Reports? | Diagnostic labels? | Research use |
|---|---|---|---|---|
| **BraTS** | Multi-institutional brain MRI (T1, T1CE, T2, FLAIR) with expert tumor segmentation masks | No free-text reports | Segmentation labels (tumor sub-regions), no per-case diagnosis text | Yes, per BraTS data use agreement |
| **IXI** | ~600 healthy-subject brain MRIs (T1, T2, PD, MRA, DWI) | No | No | Yes, freely available for research |
| **fastMRI** | Raw + reconstructed knee/brain MRI k-space data | No | No | Yes, registration required (NYU/Facebook AI) |
| **OpenNeuro** | Many open MRI/fMRI datasets, study-dependent | Varies by dataset | Varies | Yes, per individual dataset license (usually CC0) |

JSONL adapter format expected by the training/evaluation code:

```json
{"image": "case001.nii.gz", "sequence": "FLAIR", "body_region": "brain",
 "findings": ["White matter abnormality"], "report": "..."}
```

Do not include real patient identifiers in any dataset file. DICOM
metadata is anonymized by default — see [`preprocessing/dicom.py`](mri_xai_llm/preprocessing/dicom.py).
Restricted-access datasets are never auto-downloaded; the user must obtain
authorized access separately.

## 6. Model selection

See `MODEL_CONFIG` in [`config/config.py`](mri_xai_llm/config/config.py). A
pretrained medical/general vision transformer is used as the encoder
(with a torchvision ViT fallback if the medical checkpoint is
unreachable), a small instruction-tuned LLM (Qwen2.5-1.5B-Instruct, with a
Flan-T5-base fallback) handles report drafting, and a lightweight MLP
projection layer bridges the two. Segmentation uses a MONAI UNet. All
components are swappable via config — see `models/ModelManager`.

## 7. Training strategy

The vision encoder stays frozen. Only the multimodal projection layer and
(optionally) LoRA adapters on the language model are trained — full LLM
fine-tuning is intentionally out of scope for Colab-class GPUs. See
`models/multimodal_model.py: MultimodalProjector`, `apply_lora`.

## 8. Inference

```python
from main import run_pipeline
report = run_pipeline("path/to/scan.nii.gz", sequence="FLAIR", body_region="brain")
print(report.to_human_readable())
```

Or launch the full UI:

```python
from ui.gradio_app import launch_demo
launch_demo(share=True)
```

## 9. Explainable AI (XAI)

Four complementary methods are implemented (`xai/`):

- **Grad-CAM** — gradient-weighted class activation heatmaps.
- **Integrated Gradients** — via Captum, pixel-level attribution.
- **Attention Rollout** — for transformer vision encoders.
- **Occlusion Sensitivity** — measures prediction change as regions are
  masked; also powers deletion/insertion faithfulness curves.

XAI outputs are always phrased as *"the region that contributed most
strongly to the model's prediction"* — never as proof of pathology.

## 10. Evaluation

`evaluation/` provides classification metrics (accuracy, precision,
recall, F1, AUROC, AUPRC, sensitivity/specificity), segmentation metrics
(Dice, IoU, Hausdorff distance), report-generation text metrics
(BLEU/ROUGE/BERTScore — explicitly documented as **not** establishing
clinical correctness), finding-level sensitivity/specificity,
false-negative rate, and hallucination/unsupported-finding rate, plus
deletion/insertion faithfulness testing for XAI explanations.

## 11. Limitations

- Pretrained weights (e.g. `microsoft/rad-dino`) may be unreachable in an
  offline/sandboxed environment; the system falls back to general-purpose
  backbones and clearly logs when it does so.
- Confidence scores are model-derived heuristics, **not** clinically
  calibrated probabilities.
- Segmentation, lesion counts, and dimensions are approximate.
- Text-similarity metrics (BLEU/ROUGE/BERTScore) do not establish clinical
  correctness.
- This system has not been validated on any clinical population and must
  not inform real patient care.

## 12. Safety considerations

- Every report and UI screen displays the research/educational disclaimer.
- Findings below `MIN_CONFIDENCE` (config.yaml) are explicitly flagged
  "Uncertain finding — requires expert review," never hidden or upgraded.
- DICOM patient-identifying fields (name, DOB, address, hospital/accession
  ID, phone) are stripped by default and never surfaced in the UI or report.
- No cloud upload by default; no persistent patient data storage; temp
  files are cleaned up after each session.
- The pipeline never lets the LLM freely invent findings — it only
  elaborates on pre-validated, evidence-checked structured data
  (`reporting/findings.py`, hallucination-control pipeline).

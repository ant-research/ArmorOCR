<div align="center">

# ArmorOCR: Grounded Adversarial Visual Perception via Observation-Transferred Self-Distillation


<div>
    <a href="https://huggingface.co/inclusionAI/ArmorOCR"><img src="https://img.shields.io/badge/🤗 Model-Hugging Face-orange"></a>
    <a href="https://huggingface.co/inclusionAI/ArmorOCR-GGUF"><img src="https://img.shields.io/badge/🤗 GGUF-Hugging Face-orange"></a>
    <a href="https://huggingface.co/datasets/inclusionAI/advspot-public"><img src="https://img.shields.io/badge/🤗 Dataset-Hugging Face-yellow"></a>
    <a href="https://arxiv.org/abs/2608.20122"><img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
</div>

</div>

This is the official inference code of **ArmorOCR**, a two-stage framework for grounded adversarial OCR perception via observation-transferred self-distillation and reward-driven refinement. ArmorOCR is built on Qwen3-VL-8B-Instruct and enables single-pass inference on the original image, without any inference-time visual transformations or tool assistance.

<p align="center">
    <img src="figures/ocr_framework.svg" alt="ArmorOCR Framework" style="max-width:100%; height:auto;">
</p>

<div align="left">

## 🧠 Abstract

Large multimodal models (LMMs) have demonstrated strong OCR recognition capabilities, yet remain vulnerable to adversarial visual text that is readable to humans but challenging for models to localize and recognize. Existing OCR benchmarks mainly focus on natural or document-style text, while adversarial OCR evaluations remain limited in scale, task coverage, or region-aware evaluation.

In this paper, we formulate adversarial OCR as a **grounded OCR perception** task and introduce **AdvSpot**, the first benchmark for grounded adversarial OCR evaluation. AdvSpot comprises 390 images with region-level annotations, spanning 5 primary categories and 13 fine-grained adversarial OCR types. To address this challenge, we propose **ArmorOCR**, a two-stage training framework for robust adversarial OCR perception. ArmorOCR first acquires missing adversarial OCR perception from privileged transformed observations through On-Policy Self-Distillation (OPSD), and then refines grounded OCR perception through Group Relative Policy Optimization (GRPO) with task-conditioned rewards for localization, recognition, full spotting, and visual question answering (VQA).

Extensive experiments on AdvSpot, other adversarial OCR benchmarks, and general OCR benchmarks demonstrate that ArmorOCR consistently improves adversarial OCR perception while preserving competitive general OCR capability.

## 🚀 Release

- [2026/09/23] 🔥 Released the **AdvSpot benchmark data** ([inclusionAI/advspot-public](https://huggingface.co/datasets/inclusionAI/advspot-public)) and updated `advspot_infer.py` (faster prefetched data loading).
- [2026/09/01] 🔥 Released the **GGUF quantized weights** ([inclusionAI/ArmorOCR-GGUF](https://huggingface.co/inclusionAI/ArmorOCR-GGUF)).
- [2026/08/24] 🔥 Released the AdvSpot evaluation/inference script (`advspot_infer.py`).
- [2026/08/10] 🔥 Released the inference code and examples.

## 📦 AdvSpot Benchmark

AdvSpot is the first grounded adversarial OCR perception benchmark, providing:

- **383 images** with region-level annotations (bounding boxes, transcriptions, perception-type labels, region-grounded VQA pairs) → **389 grounded VQA pairs**.
> 📊 **Public release vs. paper.** The paper reports 390 images / 397 VQA pairs. This public release is a **desensitized** version (sensitive samples removed, with replacements synthesized to keep every subtype at 30–50 samples) → 383 images / 389 pairs. See the dataset card for details.
- **5 primary categories** and **13 fine-grained adversarial OCR types** organized by underlying perception failure mechanisms:
  - Spatial Manipulation (Rotated / Mirrored / Tiny Text)
  - Glyph Variation (Stylized / Handwritten Text)
  - Visual Encoding (Symbol / Dot / Line Encoding)
  - Contextual Blending (AIGC Fusion / Low Contrast / Pattern Overlay)
  - Imaging Degradation (Capture / Post-processing Artifacts)
- **Region-grounded evaluation** with VQA accuracy and IoU metrics.

<div align="center">

**Taxonomy of adversarial OCR perception in AdvSpot.**

<img src="figures/bench_taxonomy.svg" alt="AdvSpot taxonomy" style="max-width:100%; height:auto;">

**Representative adversarial OCR examples across the 13 fine-grained types.**

<img src="figures/advspot_examples.svg" alt="AdvSpot examples" style="max-width:100%; height:auto;">

</div>

The AdvSpot benchmark data is now available at 👉 [inclusionAI/advspot-public](https://huggingface.co/datasets/inclusionAI/advspot-public) (`data_public.jsonl` + `images/`; 383 images / 389 grounded VQA pairs; a desensitized public release). See [Evaluate on the AdvSpot benchmark](#3-evaluate-on-the-advspot-benchmark) for the evaluation/inference script (`advspot_infer.py`).

## ⚙️ Installation

```bash
conda create -n armorocr python=3.11
conda activate armorocr

pip install pillow==12.0.0
pip install torch==2.8.0 torchvision==0.23.0
pip install transformers==4.57.1 accelerate==1.12.0
```

> ⚠️ **Note:** ArmorOCR is built on the [Qwen3-VL](https://github.com/qwenlm/qwen3-vl) series. Please make sure your `transformers` version satisfies the minimum requirement of Qwen3-VL (see [Qwen3-VL README](https://github.com/qwenlm/qwen3-vl#requirements) for details).

## 🔍 Inference

### 1. Download model weights

Download the pre-trained weights from 👉 [inclusionAI/ArmorOCR](https://huggingface.co/inclusionAI/ArmorOCR) on Hugging Face.
> We also provide quantized GGUF checkpoints at 👉 [inclusionAI/ArmorOCR-GGUF](https://huggingface.co/inclusionAI/ArmorOCR-GGUF), along with a minimal `serve_gguf.sh` + inference example in that repo.
> The GGUF repo also includes an evaluation comparison table on AdvSpot (base vs. Q8_0 vs. Q4_K_M).

### 2. Run inference on the provided examples

```bash
python infer.py
```

Before running, please modify `YOUR_MODEL_PATH` in `infer.py` to point to your local checkpoint directory.

### 3. Evaluate on the AdvSpot benchmark

We provide `advspot_infer.py` for running ArmorOCR (and other Qwen3-VL-series models) on the AdvSpot benchmark and computing region-grounded evaluation metrics (VQA accuracy + IoU).

```bash
# 1) Download the benchmark data from https://huggingface.co/datasets/inclusionAI/advspot-public
#    (data_public.jsonl + images/) into a local directory, e.g. ./advspot-public

python advspot_infer.py \
    --model YOUR_MODEL_PATH \
    --data-file ./advspot-public/data_public.jsonl \
    --output-dir ./outputs \
    --num-gpus 1
```

Outputs include predictions (`<prefix>.json` / `<prefix>.csv`), region-grounded metrics (`<prefix>#metrics.json`), and badcases (`<prefix>#badcases.csv`). Use `--num-gpus > 1` for data-parallel inference, `--model-arch Qwen3-VL-MoE` for MoE checkpoints, and `--resume` to continue from a previous run. Run `python advspot_infer.py --help` for the full list of options.

## 🖼️ Examples

We provide four adversarial OCR examples in the `examples/` folder. Each example corresponds to a different adversarial pattern, with the model's perception analysis and final answer visualized in the corresponding case-study figure.

<div align="center">

| Example | Input | Model Output (case study) |
|:---:|:---:|:---:|
| 1 | `examples/example_1.png` | <img src="figures/case_study_1.svg" alt="case 1" style="max-width:100%; height:auto;"> |
| 2 | `examples/example_2.png` | <img src="figures/case_study_2.svg" alt="case 2" style="max-width:100%; height:auto;"> |
| 3 | `examples/example_3.png` | <img src="figures/case_study_3.svg" alt="case 3" style="max-width:100%; height:auto;"> |
| 4 | `examples/example_4.jpeg` | <img src="figures/case_study_4.svg" alt="case 4" style="max-width:100%; height:auto;"> |

</div>

## 🤝 Acknowledgement

This work would not have been possible without the following excellent projects:

1. [Qwen3-VL](https://github.com/qwenlm/qwen3-vl) — the base vision-language model family ArmorOCR is built upon.
2. [ms-swift](https://github.com/modelscope/ms-swift) — the training framework used for ArmorOCR's two-stage post-training.
3. [llama.cpp](https://github.com/ggml-org/llama.cpp) — used for the GGUF quantized release and `llama-server` inference.

## 🔗 Other Work

For readers interested in this direction, we also maintain
[clh124/Awesome-Hard-OCR-LMM](https://github.com/clh124/Awesome-Hard-OCR-LMM) — a curated,
actively-updated list of hard-OCR research in the MLLM/VLM era, spanning **benchmarks &
datasets, research papers, and competitions** across four dimensions:

- **Degraded and in-the-wild OCR** — blur, low resolution, noise, screen photos, occlusion, small/dense text.
- **Complex-document & structured OCR** — layout, tables, formulas, charts, multi-page context.
- **Script-diverse / historical / handwritten OCR** — low-resource scripts, ancient documents, handwriting.
- **Synthetic / hidden / adversarial OCR** — AIGC / synthetic text, hidden text, adversarial visual text.

AdvSpot and ArmorOCR are listed there under the last category; contributions are welcome —
open an issue on that repo to suggest a missing hard-OCR item.
</div>

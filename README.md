# Wavelet Diffusion Model for Neuroimage Foundation Modeling with FOMO45K

This repository contains the current code path for our wavelet-domain
diffusion pretraining, distilled backbone, and downstream experiments.

The repo is scoped to three downstream tasks:

- brain age regression
- FCD classification
- FCD segmentation

## Layout

- `src/pretraining`
  - wavelet diffusion pretraining and distilled student backbone
- `src/downstream`
  - fine-tuning entry point, task models, losses, and shared data code
- `pipelines/brain_age`
  - dataset construction, split building, training, inference, and result export
- `pipelines/fcd_classification`
  - ds004199 FCD classification pipeline
- `pipelines/fcd_segmentation`
  - ds004199 FCD segmentation pipeline
- `pipelines/shared`
  - shared audit utilities

## Naming conventions

Pipeline files use a small set of verb-first names:

- `build_*` for dataset or split construction
- `run_*` for Slurm launchers
- `evaluate_*` for inference-time evaluation
- `aggregate_*` for result collation
- `export_*` for workbook-style report outputs

## Preprocessing contract

The model-facing preprocessing used throughout this repo is:

- crop-to-nonzero foreground
- `RAS` orientation
- `1 mm` isotropic spacing
- `keep_aspect_ratio_when_using_target_size=False`
- `transpose=[0, 1, 2]`
- `1st/99th percentile intensity scaling to `[-1, 1]` before the wavelet transform`

That percentile scaling is the intensity normalization that matters for
describing the method, because it is the normalization applied at model input
for pretraining, distillation, fine-tuning, and inference.

For multimodal FCD tasks, FLAIR is first resampled into T1 space and masked
with the T1 brain mask before the shared geometry preprocessing and final
model-input scaling are applied.

## What is in scope

- the code needed to rebuild the current benchmark datasets
- the code needed to fine-tune and evaluate the supported downstream models
- the Slurm launchers we still need for the current journal-facing pipeline

## What stays outside the repo

- raw MRI datasets
- large preprocessed arrays
- pretrained checkpoints
- experiment logs
- QC assets and exploratory result dumps

## Reproducibility

The goal here is not to preserve every historical branch. It is to keep one
clean, traceable path:

1. build or rebuild the benchmark data
2. fine-tune the downstream model
3. run evaluation
4. aggregate or export the final results

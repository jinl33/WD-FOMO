# lightweight-fomo

`lightweight-fomo` packages the current code path for our wavelet-domain
diffusion pretraining and downstream neuroimaging experiments.

The repo is scoped to three downstream tasks:

- brain age regression
- FCD classification
- FCD segmentation

## Layout

- `src/pretraining`
  - wavelet-domain diffusion pretraining and distilled student backbone
- `src/downstream`
  - fine-tuning entry point, task models, losses, and shared data code
- `pipelines/brain_age`
  - dataset construction, split building, training, inference, and result export
- `pipelines/fcd_classification`
  - corrected ds004199 FCD classification pipeline
- `pipelines/fcd_segmentation`
  - corrected ds004199 FCD segmentation pipeline
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

Offline preprocessing is intentionally aligned across pretraining and downstream
dataset builders. The shared definition lives in
`src/downstream/data/preprocessing_defaults.py` and is used for:

- per-volume z-normalization
- crop-to-nonzero foreground
- `RAS` orientation
- `1 mm` isotropic spacing
- `keep_aspect_ratio_when_using_target_size=False`
- `transpose=[0, 1, 2]`

This means the current checkpoint-compatible pipeline uses a two-stage
normalization path:

1. offline export applies `volume_wise_znorm`
2. model input applies percentile scaling to `[-1, 1]`

That is the path used by the checkpoints and downstream results in this repo.
It is intentionally different from a pure geometry-only export pipeline because
changing the offline normalization now would no longer match the existing
pretrained weights and benchmark results.

For multimodal FCD tasks, FLAIR is first resampled into T1 space and masked
with the T1 brain mask before the shared preprocessing contract is applied.

For CleanDIFT, runtime intensity handling is also matched to pretraining: inputs
are center pad/cropped to the model patch size and then robustly scaled with the
same 1st/99th percentile mapping to `[-1, 1]` before the wavelet transform.

## What is in scope

- the code needed to rebuild the current benchmark datasets
- the code needed to fine-tune and evaluate the supported downstream models
- the Slurm launchers we still need for the current journal-facing pipeline

## What stays outside the repo

- raw MRI datasets
- large preprocessed arrays
- pretrained checkpoints
- experiment logs
- bulky QC assets and exploratory result dumps

## Reproducibility

The goal here is not to preserve every historical branch. It is to keep one
clean, traceable path:

1. build or rebuild the benchmark data
2. fine-tune the downstream model
3. run evaluation
4. aggregate or export the final results

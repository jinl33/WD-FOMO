#!/usr/bin/env python

import argparse
import os
import logging
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models.supervised_base import BaseSupervisedModel
from augmentations.finetune_augmentation_presets import (
    get_finetune_augmentation_params,
)
from utils.utils import (
    SimplePathConfig,
    setup_seed,
    find_checkpoint,
    load_pretrained_weights,
)

from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset

from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_paths import detect_version
from data.dataset import FOMODataset, ModalitySelectYuccaTrainDataset
from data.task_configs import (
    task1_config,
    task2_config,
    task3_config,
    task5_config,
    task6_config,
    task7_config,
    task8_config,
    task9_config,
    task10_config,
    task11_config,
    task12_config,
)

def get_task_config(taskid):
    if taskid == 1:
        task_cfg = task1_config
    elif taskid == 2:
        task_cfg = task2_config
    elif taskid == 3:
        task_cfg = task3_config
    elif taskid == 5:
        task_cfg = task5_config
    elif taskid == 6:
        task_cfg = task6_config
    elif taskid == 7:
        task_cfg = task7_config
    elif taskid == 8:
        task_cfg = task8_config
    elif taskid == 9:
        task_cfg = task9_config
    elif taskid == 10:
        task_cfg = task10_config
    elif taskid == 11:
        task_cfg = task11_config
    elif taskid == 12:
        task_cfg = task12_config
    else:
        raise ValueError(
            f"Unknown taskid: {taskid}. Supported IDs are 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, and 12"
        )

    return task_cfg

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory",
        default="./data/preprocessed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save models and results",
        default="./data/models",
    )
    parser.add_argument(
        "--pretrained_weights_path", type=str, help="Ckpt to finetune", default=None
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_b",
        help="Model name defined in models.networks (unet_b, unet_xl, etc.)",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument(
        "--patch_size",
        type=int,
        default=32,
        help="Patch size (cubed). Overridden by --patch_size_dhw if provided.",
    )
    parser.add_argument(
        "--patch_size_dhw",
        type=int,
        nargs=3,
        default=None,
        help="Explicit (D, H, W) patch size. If set, overrides --patch_size.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--backbone_learning_rate",
        type=float,
        default=None,
        help=(
            "Optional lower learning rate for pretrained CleanDIFT backbone "
            "parameters. The main --learning_rate is used for task-specific "
            "heads, fusion layers, and decoders."
        ),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "none"],
        default="basic",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_batches_per_epoch", type=int, default=100)
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before weight update",
    )
    parser.add_argument(
        "--overfit_batches",
        type=int,
        default=0,
        help="Number of batches to overfit on. If >0, disables shuffling and limit_train_batches.",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.0,
        help="Gradient clipping value (0 = disabled). Recommended: 1.0 for stability.",
    )
    parser.add_argument(
        "--regression_loss",
        type=str,
        choices=["mse", "smoothl1"],
        default=None,
        help="Override regression loss for Task 3 / brain-age runs.",
    )
    parser.add_argument(
        "--regression_smoothl1_beta",
        type=float,
        default=10.0,
        help="Beta parameter when regression_loss=smoothl1.",
    )
    parser.add_argument(
        "--regression_rank_loss_weight",
        type=float,
        default=0.0,
        help="Optional weight for pairwise age-ranking regularization.",
    )
    parser.add_argument(
        "--segmentation_loss",
        type=str,
        choices=[
            "dicece",
            "dicece_sizeaware",
            "dicece_nobg",
            "dicece_nobg_wce",
            "focaltversky",
            "focaltverskyce",
            "focaltverskyce_wce",
            "generalizeddice",
            "generalizeddicece",
            "tverskyce",
        ],
        default=None,
        help="Override segmentation loss for Tasks 2 / 10.",
    )
    parser.add_argument(
        "--seg_attention",
        type=str,
        choices=["none", "se"],
        default="none",
        help="Optional segmentation decoder attention variant for CleanDIFT.",
    )
    parser.add_argument(
        "--seg_head_variant",
        type=str,
        choices=[
            "earlyfusion",
            "waveletavg",
            "waveletweighted",
            "waveletavg_refine",
            "waveletweighted_refine",
            "image_refine_v1",
            "image_refine_ms",
            "image_refine_ms_aux",
            "featureavg",
            "latefusion_logits",
            "flairskip_latefusion",
            "modality_expert",
            "modality_expert_softmax",
            "modality_expert_avg",
            "modality_expert_logitconv",
            "modality_expert_featureconcat",
            "modality_expert_sharedbackbone",
            "image_refine",
            "cascade_refine",
            "image_refine_ptdec",
            "waveletavg_refine_ptdec",
        ],
        default="earlyfusion",
        help="Segmentation-head variant for CleanDIFT dense prediction.",
    )
    parser.add_argument(
        "--seg_output_fg_prior",
        type=float,
        default=None,
        help=(
            "Optional binary foreground prior for initializing the CleanDIFT "
            "segmentation output bias."
        ),
    )
    parser.add_argument(
        "--seg_aux_loss_weight",
        type=float,
        default=0.0,
        help="Optional auxiliary boundary-loss weight for CleanDIFT segmentation heads.",
    )
    parser.add_argument(
        "--seg_aux_target",
        type=str,
        choices=["binary_band", "distance_shell"],
        default="binary_band",
        help="Target style for the CleanDIFT auxiliary boundary branch.",
    )
    parser.add_argument(
        "--seg_boundary_radius",
        type=int,
        default=1,
        help="Morphological radius used to define auxiliary boundary supervision.",
    )
    parser.add_argument(
        "--seg_small_lesion_weight",
        type=float,
        default=0.0,
        help=(
            "Extra per-voxel CE weight applied to voxels that belong to small "
            "lesions when segmentation_loss=dicece_sizeaware."
        ),
    )
    parser.add_argument(
        "--seg_small_lesion_thresholds",
        type=int,
        nargs=3,
        default=[1000, 10000, 50000],
        metavar=("SMALL", "MEDIUM", "LARGE"),
        help=(
            "Voxel-count thresholds for size-aware lesion weighting. Lesions up "
            "to SMALL get the full extra weight, then taper through MEDIUM and LARGE."
        ),
    )
    parser.add_argument(
        "--p_oversample_foreground",
        type=float,
        default=0.33,
        help="Probability that a segmentation training patch is forced to include foreground.",
    )
    parser.add_argument(
        "--input_modality_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional zero-based modality indices to use from a preprocessed "
            "multi-modal case. Example for Task10 FLAIR-only: --input_modality_indices 1."
        ),
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Enable early stopping when >0 (patience in epochs).",
    )
    parser.add_argument(
        "--early_stopping_monitor",
        type=str,
        default="val/loss",
        help="Metric name to monitor for early stopping.",
    )
    parser.add_argument(
        "--early_stopping_mode",
        type=str,
        choices=["min", "max"],
        default="min",
        help="Optimization direction for early stopping monitor.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum absolute improvement to reset patience.",
    )
    parser.add_argument(
        "--best_checkpoint_monitor",
        type=str,
        default="val/loss",
        help="Metric name to monitor when selecting best.ckpt.",
    )
    parser.add_argument(
        "--best_checkpoint_mode",
        type=str,
        choices=["min", "max"],
        default="min",
        help="Optimization direction for best checkpoint selection.",
    )
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        default="full",
        help=(
            "Optional finetune strategy override. "
            "Supported: full, cleandift_lateblocks, cleandift_peft."
        ),
    )
    parser.add_argument(
        "--linear_probing",
        action="store_true",
        help="Freeze the pretrained backbone and train only the task head (plus tiny input adapters when needed).",
    )

    parser.add_argument(
        "--taskid",
        type=int,
        required=True,
        help=(
            "Task ID (1: FOMO1 classification, 2: FOMO2 segmentation, 3: FOMO3 regression, "
            "5/6/7: FOMO300K brain-age variants, 8: legacy ds004199 export, "
            "9: corrected ds004199 T1+FLAIR FCD classification, "
            "10: corrected ds004199 T1+FLAIR FCD lesion segmentation, "
            "11: BrainLat T1 SynthSeg anatomical warm-up)"
        ),
    )
    parser.add_argument(
        "--split_method", type=str, default="stratified_train_val_split"
    )
    parser.add_argument(
        "--split_param",
        type=str,
        help="Split parameter (e.g., 0.2 for simple, 5 for kfold)",
        default="0.2",
    )
    parser.add_argument(
        "--split_idx", type=int, default=0, help="Index of the split to use for kfold"
    )
    parser.add_argument(
        "--experiment", type=str, default="experiment", help="name of experiment"
    )
    args = parser.parse_args()

    if args.patch_size_dhw is not None:
        patch_size_tuple = tuple(args.patch_size_dhw)
    else:
        patch_size_tuple = (args.patch_size,) * 3

    if args.model_name.startswith("cleandift"):
        bad = [ps for ps in patch_size_tuple if ps % 8 != 0]
        if bad:
            print(
                "Allowing non-divisible patch dims for CleanDIFT because the "
                "wavelet backbone pads internally before feature extraction: "
                f"{patch_size_tuple}"
            )
    else:
        for ps in patch_size_tuple:
            assert ps % 8 == 0, f"Patch size dims must be divisible by 8, got {ps}"

    task_cfg = get_task_config(args.taskid)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    num_classes = task_cfg["num_classes"]
    task_modalities = task_cfg["modalities"]
    if args.input_modality_indices is not None:
        bad = [
            idx
            for idx in args.input_modality_indices
            if idx < 0 or idx >= len(task_modalities)
        ]
        if bad:
            raise ValueError(
                f"Requested modality indices {bad}, but task {task_name} has "
                f"{len(task_modalities)} modalities: {task_modalities}"
            )
        modalities = len(args.input_modality_indices)
        selected_modalities = tuple(task_modalities[idx] for idx in args.input_modality_indices)
        os.environ["FOMO_INPUT_MODALITY_INDICES"] = ",".join(
            str(idx) for idx in args.input_modality_indices
        )
        print(
            "Using modality subset: "
            f"indices={args.input_modality_indices}, names={selected_modalities}"
        )
    else:
        modalities = len(task_modalities)
        selected_modalities = task_modalities
        os.environ.pop("FOMO_INPUT_MODALITY_INDICES", None)
    labels = task_cfg["labels"]

    run_type = "from_scratch" if args.pretrained_weights_path is None else "finetune"
    experiment_name = f"{run_type}_{args.model_name}_{args.experiment}_{args.taskid}"

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print(f"Task type: {task_type}")
    print("ARGS:", args)

    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, task_name)

    save_dir = os.path.join(args.save_dir, task_name, args.model_name)

    continue_from_most_recent = not args.new_version
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    if "kfold" in args.split_method:
        split_param = int(args.split_param)
    elif args.split_method in [
        "simple_train_val_split",
        "stratified_train_val_split",
        "stratified_train_val_test_split",
    ]:
        split_param = float(args.split_param)
    else:
        split_param = args.split_param

    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_split_config(
        method=args.split_method,
        param=split_param,
        path_config=path_config,
    )

    seed = setup_seed(continue_from_most_recent)
    ckpt_path = find_checkpoint(version_dir, continue_from_most_recent)

    effective_batch_size = (
        args.num_devices * args.batch_size * args.accumulate_grad_batches
    )
    train_dataset_size = len(splits_config.train(args.split_idx))
    val_dataset_size = len(splits_config.val(args.split_idx))
    max_iterations = int(args.epochs * args.train_batches_per_epoch)

    config = {
        "task": task_name,
        "task_id": args.taskid,
        "task_type": task_type,
        "experiment": experiment_name,
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "run_type": run_type,
        "split_method": args.split_method,
        "split_param": split_param,
        "split_idx": args.split_idx,
        "save_dir": save_dir,
        "train_data_dir": train_data_dir,
        "version_dir": version_dir,
        "version": version,
        "ckpt_path": ckpt_path,
        "pretrained_weights_path": args.pretrained_weights_path,
        "seed": seed,
        "num_classes": num_classes,
        "num_modalities": modalities,
        "input_modality_indices": args.input_modality_indices,
        "selected_modalities": selected_modalities,
        "image_extension": ".npy",
        "allow_missing_modalities": False,
        "labels": labels,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "patch_size": patch_size_tuple,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": effective_batch_size,
        "regression_loss": args.regression_loss,
        "regression_smoothl1_beta": args.regression_smoothl1_beta,
        "regression_rank_loss_weight": args.regression_rank_loss_weight,
        "segmentation_loss": args.segmentation_loss,
        "seg_attention": args.seg_attention,
        "seg_head_variant": args.seg_head_variant,
        "seg_output_fg_prior": args.seg_output_fg_prior,
        "seg_aux_loss_weight": args.seg_aux_loss_weight,
        "seg_aux_target": args.seg_aux_target,
        "seg_boundary_radius": args.seg_boundary_radius,
        "seg_small_lesion_weight": args.seg_small_lesion_weight,
        "seg_small_lesion_thresholds": args.seg_small_lesion_thresholds,
        "p_oversample_foreground": args.p_oversample_foreground,
        "train_dataset_size": train_dataset_size,
        "val_dataset_size": val_dataset_size,
        "max_iterations": max_iterations,
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "fast_dev_run": args.fast_dev_run,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_monitor": args.early_stopping_monitor,
        "early_stopping_mode": args.early_stopping_mode,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_checkpoint_monitor": args.best_checkpoint_monitor,
        "best_checkpoint_mode": args.best_checkpoint_mode,
        "finetune_strategy": args.finetune_strategy,
        "linear_probing": args.linear_probing,
    }

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename="{epoch}",
        enable_version_counter=False,
    )
    best_checkpoint_callback = ModelCheckpoint(
        monitor=args.best_checkpoint_monitor,
        mode=args.best_checkpoint_mode,
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    callbacks = [checkpoint_callback, best_checkpoint_callback]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stopping_monitor,
                mode=args.early_stopping_mode,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                verbose=True,
            )
        )

    yucca_logger = YuccaLogger(
        save_dir=save_dir,
        version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )

    loggers = [yucca_logger]

    aug_params = get_finetune_augmentation_params(args.augmentation_preset)
    tt_preset = "classification" if task_type == "regression" else task_type
    augmenter = YuccaAugmentationComposer(
        patch_size=config["patch_size"],
        task_type_preset=tt_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    data_module = YuccaDataModule(
        train_dataset_class=(
            ModalitySelectYuccaTrainDataset
            if task_type == "segmentation"
            else FOMODataset
        ),
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        train_data_dir=config["train_data_dir"],
        image_extension=config["image_extension"],
        task_type=config["task_type"],
        splits_config=splits_config,
        split_idx=config["split_idx"],
        num_workers=args.num_workers,
        p_oversample_foreground=args.p_oversample_foreground,
        val_sampler=None,
    )

    print("Train dataset: ", data_module.splits_config.train(config["split_idx"]))
    print("Val dataset: ", data_module.splits_config.val(config["split_idx"]))
    print("Run type: ", run_type)
    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with train dataset of size {train_dataset_size} datapoints and val dataset of size {val_dataset_size} "
        f"and effective batch size of {effective_batch_size}"
    )

    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=args.learning_rate,
        do_compile=args.compile,
        compile_mode="default" if args.compile_mode is None else args.compile_mode,
    )

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=save_dir,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch
        if args.overfit_batches == 0
        else None,
        overfit_batches=args.overfit_batches if args.overfit_batches > 0 else 0.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=args.gradient_clip_val
        if args.gradient_clip_val > 0
        else None,
    )

    if run_type == "finetune":
        print("Transferring weights for finetuning")
        print(f"Checkpoint path: {ckpt_path}")
        assert ckpt_path is None, (
            "Error: You're attempting to load pretrained weights while "
            "simultaneously continuing from a checkpoint. This creates "
            "conflicting weight sources. Use either --pretrained_weights_path "
            "for finetuning OR continue training without the --new_version flag, "
            "but not both."
        )
        state_dict = load_pretrained_weights(args.pretrained_weights_path, args.compile)

        if args.model_name.startswith("cleandift"):
            state_expander = getattr(model.model, "expand_pretrained_state_dict", None)
            downstream_ckpt = any(
                k.startswith("model.backbone.") or k.startswith("model.decoder.")
                for k in state_dict.keys()
            )
            if state_expander is not None:
                state_dict = state_expander(state_dict, prefix="model.")
                print(
                    "Expanded CleanDIFT pretrained backbone keys for "
                    f"modality-expert branches: {len(state_dict)} keys"
                )
            elif downstream_ckpt:
                compatible = {}
                model_state = model.state_dict()
                backbone_n = 0
                decoder_n = 0
                other_n = 0
                for k, v in state_dict.items():
                    if k.startswith("model.modality_fusion."):
                        continue
                    if k.startswith("model.decoder.final."):
                        continue
                    if k in model_state and model_state[k].shape == v.shape:
                        compatible[k] = v
                        if k.startswith("model.backbone."):
                            backbone_n += 1
                        elif k.startswith("model.decoder."):
                            decoder_n += 1
                        else:
                            other_n += 1
                state_dict = compatible
                print(
                    "Loaded downstream CleanDIFT warm-up checkpoint: "
                    f"{len(compatible)} compatible keys "
                    f"(backbone={backbone_n}, decoder={decoder_n}, other={other_n})"
                )
            else:
                remapped = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_key = k.replace("model.", "model.backbone.", 1)
                        remapped[new_key] = v
                    elif k.startswith("teacher.") or k.startswith("projection_heads."):
                        continue
                    else:
                        remapped["model.backbone." + k] = v
                state_dict = remapped
                print(f"Remapped {len(remapped)} CleanDIFT backbone keys")

        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        in_ckpt_not_model = ckpt_keys - model_keys
        in_model_not_ckpt = model_keys - ckpt_keys
        print(
            f"  Ckpt keys not in model ({len(in_ckpt_not_model)}): "
            f"{sorted(in_ckpt_not_model)[:5]}{'...' if len(in_ckpt_not_model)>5 else ''}"
        )
        print(
            f"  Model keys not in ckpt ({len(in_model_not_ckpt)}): "
            f"{sorted(in_model_not_ckpt)[:5]}{'...' if len(in_model_not_ckpt)>5 else ''}"
        )

        num_successful_weights_transferred = model.load_state_dict(
            state_dict=state_dict, strict=False
        )
        assert (
            num_successful_weights_transferred > 0
        ), "No weights were successfully transferred"
    else:
        print("Training from scratch, no weights will be transferred")

    trainer.fit(model=model, datamodule=data_module, ckpt_path="last")

if __name__ == "__main__":
    main()

from typing import Optional
import torch
from torch.optim import AdamW
import copy
import logging

import lightning as L
from yucca.pipeline.preprocessing import YuccaPreprocessor
from yucca.functional.utils.kwargs import filter_kwargs
from batchgenerators.utilities.file_and_folder_operations import join
from models import networks

class BaseSupervisedModel(L.LightningModule):

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.config = config
        self.num_classes = config["num_classes"]
        self.num_modalities = config["num_modalities"]
        self.patch_size = config["patch_size"]
        self.plans = config.get("plans", {})
        self.model_name = config["model_name"]
        self.version_dir = config["version_dir"]
        self.task_type = config["task_type"]
        self.run_type = config.get("run_type", "from_scratch")
        self.finetune_strategy = config.get("finetune_strategy", "full")
        self.linear_probing = bool(config.get("linear_probing", False))

        self.sliding_window_prediction = self.task_type != "regression"
        self.sliding_window_overlap = 0.5
        self.test_time_augmentation = False
        self.progress_bar = True

        self.do_compile = do_compile
        self.compile_mode = compile_mode

        self.deep_supervision = deep_supervision

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.eps = eps
        self.betas = betas

        self.train_metrics = self._configure_metrics(prefix="train")
        self.val_metrics = self._configure_metrics(prefix="val")

        self.save_hyperparameters()
        self.load_model()

        self.model = (
            torch.compile(self.model, mode=self.compile_mode)
            if self.do_compile
            else self.model
        )

    def _configure_metrics(self, prefix: str):
        raise NotImplementedError("Subclasses must implement _configure_metrics")

    def _configure_losses(self):
        raise NotImplementedError("Subclasses must implement _configure_losses")

    def load_model(self):
        print(f"Loading Model: 3D {self.model_name}")
        model_class = getattr(networks, self.model_name)

        print("Found model class: ", model_class)

        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d
        print("MODALITIES", self.num_modalities)

        model_kwargs = {
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            "output_channels": self.num_classes,
            "deep_supervision": self.deep_supervision,
            "conv_op": conv_op,
            "norm_op": norm_op,
            "checkpoint_style": None,
            "mode": self.task_type,
            "use_vae": False,
            "use_skip_connections": False,
            "seg_attention": self.config.get("seg_attention", "none"),
            "seg_head_variant": self.config.get("seg_head_variant", "earlyfusion"),
            "seg_output_fg_prior": self.config.get("seg_output_fg_prior"),
        }
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        if self.model_name.startswith("cleandift"):
            model_kwargs["seg_attention"] = self.config.get("seg_attention", "none")
            model_kwargs["seg_head_variant"] = self.config.get(
                "seg_head_variant", "earlyfusion"
            )
            model_kwargs["seg_output_fg_prior"] = self.config.get(
                "seg_output_fg_prior"
            )
        self.model = model_class(**model_kwargs)
        if self.model_name.startswith("cleandift") and self.task_type == "segmentation":
            expected_variant = self.config.get("seg_head_variant", "earlyfusion")
            actual_variant = getattr(self.model, "seg_head_variant", None)
            print("CleanDIFT segmentation head variant:", actual_variant)
            if actual_variant != expected_variant:
                raise RuntimeError(
                    "CleanDIFT segmentation head variant mismatch: "
                    f"expected={expected_variant}, actual={actual_variant}"
                )

    def configure_optimizers(self):
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

        backbone_learning_rate = self.config.get("backbone_learning_rate")
        use_backbone_group = (
            backbone_learning_rate is not None
            and self.model_name.startswith("cleandift")
            and self.task_type == "segmentation"
            and not self.linear_probing
        )

        if use_backbone_group:
            backbone_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("backbone.") or (
                    name.startswith("modality_experts.") and ".backbone." in name
                ):
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            param_groups = []
            if backbone_params:
                param_groups.append(
                    {"params": backbone_params, "lr": float(backbone_learning_rate)}
                )
            if head_params:
                param_groups.append({"params": head_params, "lr": self.learning_rate})
            if not param_groups:
                raise RuntimeError("No trainable parameters found for optimizer")

            print(
                "Using CleanDIFT segmentation differential LR: "
                f"backbone={backbone_learning_rate} "
                f"({len(backbone_params)} tensors), "
                f"head={self.learning_rate} ({len(head_params)} tensors)"
            )
            self.optim = AdamW(
                param_groups,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
                eps=self.eps,
                betas=self.betas,
            )
        else:
            self.optim = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
                eps=self.eps,
                betas=self.betas,
            )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=int(self.trainer.max_epochs * 1.15), eta_min=1e-9
        )

        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def forward(self, inputs):
        return self.model(inputs)

    def _process_batch(self, batch):
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        return inputs, target, file_path

    def training_step(self, batch, _batch_idx):
        input_channels = batch["image"].shape[1]
        assert input_channels == self.num_modalities, (
            f"Expected {self.num_modalities} input channels, but got {input_channels}. "
            "This often happens when the task config has changed _after_ data has been preprocessed. "
            "Check your data preprocessing."
        )

        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision and hasattr(output, "__iter__"):
            output = output[0]
            target = target[0]

        result = self.compute_metrics(self.train_metrics, output, target)
        if result is not None:
            self.log_dict(
                {"train/loss": loss} | result,
                prog_bar=self.progress_bar,
                logger=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        else:
            self.log(
                "train/loss",
                loss,
                prog_bar=self.progress_bar,
                logger=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                self.train_metrics,
                prog_bar=self.progress_bar,
                logger=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch, _batch_idx):
        input_channels = batch["image"].shape[1]
        assert input_channels == self.num_modalities, (
            f"Expected {self.num_modalities} input channels, but got {input_channels}. "
            "This often happens when the task config has changed _after_ data has been preprocessed. "
            "Check your data preprocessing."
        )

        inputs, target, _ = self._process_batch(batch)

        if self.model_name == "mmunetvae" and self.task_type != "segmentation":
            output = self.run_predict(inputs.float())
        else:
            output = self(inputs)
        loss = self.loss_fn_val(output, target)
        result = self.compute_metrics(self.val_metrics, output, target)
        if result is not None:
            self.log_dict(
                {"val/loss": loss} | result,
                prog_bar=self.progress_bar,
                logger=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
        else:
            self.log(
                "val/loss",
                loss,
                prog_bar=self.progress_bar,
                logger=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log_dict(
                self.val_metrics,
                prog_bar=self.progress_bar,
                logger=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_model_for_finetuning(self):
        if self.linear_probing:
            print("Applying linear probing strategy...")
            for param in self.model.parameters():
                param.requires_grad = False

            trainable_prefixes = []
            trainable_exact = set()

            if self.model_name.startswith("cleandift_"):
                trainable_prefixes.extend(
                    [
                        "head",
                        "decoder",
                        "tap_projections",
                        "tap_adapters",
                        "fused_norm",
                        "decoder_probe",
                        "seg_head",
                        "logit_fusion",
                        "mid_fuse",
                        "up0_fuse",
                        "flair_mid_residual",
                        "image_refine",
                    ]
                )
                trainable_exact.add("tap_logits")
                modality_fusion = getattr(self.model, "modality_fusion", None)
                if modality_fusion is not None and not isinstance(
                    modality_fusion, torch.nn.Identity
                ):
                    trainable_prefixes.append("modality_fusion")
                fusion_adapter = getattr(self.model, "fusion_adapter", None)
                if fusion_adapter is not None:
                    trainable_prefixes.append("fusion_adapter")
                if hasattr(self.model, "modality_experts"):
                    trainable_prefixes.extend(
                        [
                            "modality_logit_weights",
                            "modality_logit_fusion",
                            "modality_feature_fusion",
                        ]
                    )
            elif self.model_name == "mmunetvae":
                if self.task_type == "segmentation":
                    trainable_prefixes.extend(["decoder"])
                else:
                    trainable_prefixes.extend(["decoder_task"])
            else:
                trainable_prefixes.extend(["decoder"])

            for name, param in self.model.named_parameters():
                if name in trainable_exact or any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in trainable_prefixes
                ):
                    param.requires_grad = True
                if name.startswith("modality_experts."):
                    if ".backbone." in name or ".wavelet." in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            trainable = [
                name for name, p in self.model.named_parameters() if p.requires_grad
            ]
            frozen = [
                name for name, p in self.model.named_parameters() if not p.requires_grad
            ]
            print(f"Linear probe trainable params: {len(trainable)} tensors")
            if trainable:
                print(f"  sample trainable: {trainable[:10]}")
            print(f"Linear probe frozen params: {len(frozen)} tensors")
            return

        if self.model_name != "mmunetvae":
            if (
                self.model_name.startswith("cleandift_")
                and self.finetune_strategy == "cleandift_lateblocks"
            ):
                print("Applying CleanDIFT late-block finetune strategy...")
                for param in self.model.parameters():
                    param.requires_grad = False

                for module_name in ("modality_fusion", "head"):
                    module = getattr(self.model, module_name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = True

                backbone = getattr(self.model, "backbone", None)
                if backbone is not None:
                    for attr in ("middle_block",):
                        module = getattr(backbone, attr, None)
                        if module is not None:
                            for param in module.parameters():
                                param.requires_grad = True
                    down_blocks = getattr(backbone, "down_blocks", None)
                    if down_blocks:
                        for param in down_blocks[-1].parameters():
                            param.requires_grad = True

                trainable = [
                    name for name, p in self.model.named_parameters() if p.requires_grad
                ]
                print(
                    f"CleanDIFT late-block strategy trainable params: {len(trainable)} tensors"
                )
                return
            if (
                self.model_name.startswith("cleandift_")
                and self.finetune_strategy == "cleandift_peft"
            ):
                print("Applying CleanDIFT PEFT finetune strategy...")
                for param in self.model.parameters():
                    param.requires_grad = False

                trainable_prefixes = (
                    "modality_fusion",
                    "fusion_adapter",
                    "tap_adapters",
                    "tap_projections",
                    "fused_norm",
                    "head",
                )
                for name, param in self.model.named_parameters():
                    if name == "tap_logits" or name.startswith(trainable_prefixes):
                        param.requires_grad = True

                trainable = [
                    name for name, p in self.model.named_parameters() if p.requires_grad
                ]
                print(
                    f"CleanDIFT PEFT strategy trainable params: {len(trainable)} tensors"
                )
                return
            return
        if self.task_type == "segmentation":
            return
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return
        print("Freezing MMUNetVAE decoder parameters for finetuning...")
        for param in decoder.parameters():
            param.requires_grad = False

    def on_fit_start(self):
        if self.run_type == "finetune":
            self.configure_model_for_finetuning()

    def on_predict_start(self):
        self.preprocessor = YuccaPreprocessor(join(self.version_dir, "hparams.yaml"))

    def run_predict(self, inputs):
        if self.model_name == "mmunetvae" and hasattr(self.model, "predict"):
            with torch.autocast("cuda", enabled=False):
                return self.model.predict(
                    data=inputs.float(),
                    mode="3D",
                    mirror=False,
                    overlap=self.sliding_window_overlap,
                    patch_size=self.patch_size,
                    sliding_window_prediction=True,
                    device=inputs.device,
                )
        return self(inputs)

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        case, case_id = batch
        (
            case_preprocessed,
            case_properties,
        ) = self.preprocessor.preprocess_case_for_inference(
            case, self.patch_size, self.sliding_window_prediction
        )

        predictions = self.model.predict(
            data=case_preprocessed,
            mode="3D",
            mirror=self.test_time_augmentation,
            overlap=self.sliding_window_overlap,
            patch_size=self.patch_size,
            sliding_window_prediction=self.sliding_window_prediction,
            device=self.device,
        )
        predictions, case_properties = self.preprocessor.reverse_preprocessing(
            predictions, case_properties
        )
        return {
            "predictions": predictions,
            "properties": case_properties,
            "case_id": case_id[0],
        }

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        raise NotImplementedError("Subclasses must implement compute_metrics")

    def load_state_dict(self, state_dict, *args, **kwargs):
        old_params = copy.deepcopy(self.state_dict())
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if (k in old_params) and (old_params[k].shape == state_dict[k].shape)
        }
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        rejected_keys_shape = [
            k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape
        ]
        rejected_keys_data = []

        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(
            old_params.keys(), old_params.values(), new_params.values()
        ):
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if (
                    param_name not in rejected_keys_new
                    and param_name not in rejected_keys_shape
                ):
                    rejected_keys_data.append(param_name)

        logging.warn(
            f"Succesfully transferred weights for {successful}/{successful+unsuccessful} layers"
        )
        logging.warn(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )

        return successful

    @staticmethod
    def create(task_type, config, **kwargs):
        if task_type == "segmentation":
            from models.supervised_seg import SupervisedSegModel

            return SupervisedSegModel(config=config, **kwargs)
        elif task_type == "classification":
            from models.supervised_cls import SupervisedClsModel

            return SupervisedClsModel(config=config, **kwargs)
        elif task_type == "regression":
            from models.supervised_reg import SupervisedRegModel

            return SupervisedRegModel(config=config, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

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
    """
    Base class for supervised models (segmentation, classification, regression).
    Implements common functionality and defines abstract methods that subclasses must implement.
    """

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
        self.task_type = config["task_type"]  # Added task_type property
        self.run_type = config.get("run_type", "from_scratch")
        self.finetune_strategy = config.get("finetune_strategy", "full")
        self.linear_probing = bool(config.get("linear_probing", False))

        # Bug B fix: Disable sliding window for regression to avoid wraparound bug
        # When dims like [33,63] produce negative sliding window steps, PyTorch
        # tensor slicing interprets these as wrap-around indices, yielding tiny patches
        # that crash the encoder's pooling layer with "Output size too small" error.
        # Full-image prediction uses adaptive_avg_pool3d and handles all spatial sizes safely.
        self.sliding_window_prediction = (self.task_type != "regression")
        self.sliding_window_overlap = 0.5  # nnUNet default
        self.test_time_augmentation = False
        self.progress_bar = True

        self.do_compile = do_compile
        self.compile_mode = compile_mode

        # Loss
        self.deep_supervision = deep_supervision

        # Optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.eps = eps
        self.betas = betas

        # Set up metrics in subclasses
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
        """
        Configure metrics specific to the task type.
        Must be implemented by subclasses.

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of metrics for the task
        """
        raise NotImplementedError("Subclasses must implement _configure_metrics")

    def _configure_losses(self):
        """
        Configure loss functions specific to the task type.
        Must be implemented by subclasses.

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        raise NotImplementedError("Subclasses must implement _configure_losses")

    def load_model(self):
        """Load the appropriate model architecture"""
        print(f"Loading Model: 3D {self.model_name}")
        model_class = getattr(networks, self.model_name)

        print("Found model class: ", model_class)

        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d
        print("MODALITIES", self.num_modalities)

        # Pass task_type directly to UNet without mapping
        model_kwargs = {
            # Applies to all models
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            "output_channels": self.num_classes,
            "deep_supervision": self.deep_supervision,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # MedNeXt
            "checkpoint_style": None,
            # ensure not pretraining
            "mode": self.task_type,  # Pass task_type directly
            # Match the upstream fomo25 finetune path for MMUNetVAE:
            # use deterministic latent features instead of VAE sampling.
            "use_vae": False,
            "use_skip_connections": False,
        }
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        self.model = model_class(**model_kwargs)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Set up task-specific loss functions
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

        self.optim = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            eps=self.eps,
            betas=self.betas,
        )

        # Scheduler with early cut-off factor of 1.15
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=int(self.trainer.max_epochs * 1.15), eta_min=1e-9
        )

        # Return the optimizer and scheduler - the loss is not returned
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def forward(self, inputs):
        """Forward pass through the model"""
        return self.model(inputs)

    def _process_batch(self, batch):
        """Process batch data - can be overridden by subclasses if needed"""
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        return inputs, target, file_path

    def training_step(self, batch, _batch_idx):
        """Training step"""
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
            # If deep_supervision is enabled, output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        result = self.compute_metrics(self.train_metrics, output, target)
        if result is not None:
            # Subclass returned a dict of step values (e.g. segmentation
            # per-class expansion).  Log them the traditional way.
            self.log_dict(
                {"train/loss": loss} | result,
                prog_bar=self.progress_bar,
                logger=True, sync_dist=True,
                on_step=True, on_epoch=True,
            )
        else:
            # Subclass only called metrics.update() — log Metric objects
            # directly so Lightning calls .compute() on the accumulated
            # epoch state instead of averaging per-step scalars.
            # Critical for AUROC and PearsonCorrCoef.
            self.log("train/loss", loss, prog_bar=self.progress_bar,
                     logger=True, sync_dist=True, on_step=True, on_epoch=True)
            self.log_dict(self.train_metrics, prog_bar=self.progress_bar,
                          logger=True, sync_dist=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, _batch_idx):
        """Validation step"""
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
                logger=True, sync_dist=True,
                on_step=False, on_epoch=True,
            )
        else:
            self.log("val/loss", loss, prog_bar=self.progress_bar,
                     logger=True, sync_dist=True, on_step=False, on_epoch=True)
            self.log_dict(self.val_metrics, prog_bar=self.progress_bar,
                          logger=True, sync_dist=True, on_step=False, on_epoch=True)

    def configure_model_for_finetuning(self):
        """
        Match the upstream MMUNetVAE finetune path by freezing the reconstruction
        decoder while keeping the regression head trainable.
        """
        if self.linear_probing:
            print("Applying linear probing strategy...")
            for param in self.model.parameters():
                param.requires_grad = False

            trainable_prefixes = []
            trainable_exact = set()

            if self.model_name.startswith("cleandift_"):
                # Keep the pretrained diffusion backbone frozen. For multimodal
                # tasks, allow the small learned fusion layer to adapt because
                # it is task-input-specific rather than part of the pretrained
                # denoising backbone.
                trainable_prefixes.extend(
                    [
                        "head",
                        "decoder",
                        "tap_projections",
                        "tap_adapters",
                        "fused_norm",
                        "decoder_probe",
                        "seg_head",
                    ]
                )
                trainable_exact.add("tap_logits")
                modality_fusion = getattr(self.model, "modality_fusion", None)
                if modality_fusion is not None and not isinstance(modality_fusion, torch.nn.Identity):
                    trainable_prefixes.append("modality_fusion")
                fusion_adapter = getattr(self.model, "fusion_adapter", None)
                if fusion_adapter is not None:
                    trainable_prefixes.append("fusion_adapter")
            elif self.model_name == "mmunetvae":
                if self.task_type == "segmentation":
                    trainable_prefixes.extend(["decoder"])
                else:
                    trainable_prefixes.extend(["decoder_task"])
            else:
                # For UNet/MedNeXt style classification-regression models, the
                # task-specific predictor lives under `decoder`.
                trainable_prefixes.extend(["decoder"])

            for name, param in self.model.named_parameters():
                if name in trainable_exact or any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in trainable_prefixes
                ):
                    param.requires_grad = True

            trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
            frozen = [name for name, p in self.model.named_parameters() if not p.requires_grad]
            print(f"Linear probe trainable params: {len(trainable)} tensors")
            if trainable:
                print(f"  sample trainable: {trainable[:10]}")
            print(f"Linear probe frozen params: {len(frozen)} tensors")
            return

        if self.model_name != "mmunetvae":
            # Optional late-block strategy for CleanDIFT scalar finetuning.
            if self.model_name.startswith("cleandift_") and self.finetune_strategy == "cleandift_lateblocks":
                print("Applying CleanDIFT late-block finetune strategy...")
                # Freeze everything first.
                for param in self.model.parameters():
                    param.requires_grad = False

                # Always train the small task-specific pieces.
                for module_name in ("modality_fusion", "head"):
                    module = getattr(self.model, module_name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = True

                # Unfreeze the deepest trainable diffusion blocks that directly
                # shape the bottleneck features consumed by the scalar head.
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

                trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
                print(f"CleanDIFT late-block strategy trainable params: {len(trainable)} tensors")
                return
            if self.model_name.startswith("cleandift_") and self.finetune_strategy == "cleandift_peft":
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

                trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
                print(f"CleanDIFT PEFT strategy trainable params: {len(trainable)} tensors")
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
        """Set up for prediction"""
        self.preprocessor = YuccaPreprocessor(join(self.version_dir, "hparams.yaml"))

    def run_predict(self, inputs):
        """
        Upstream-style validation path for MMUNetVAE. For other models, fall back
        to the regular forward pass.
        """
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
        """Prediction step"""
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
        """
        Compute task-specific metrics.
        Should be implemented/extended by subclasses for task-specific metrics.
        """
        raise NotImplementedError("Subclasses must implement compute_metrics")

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict with handling for different model architectures"""
        # First we filter out layers that have changed in size
        # This is often the case in the output layer.
        # If we are finetuning on a task with a different number of classes
        # than the pretraining task, the # output channels will have changed.
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

        # Here there's also potential to implement custom loading functions.
        # E.g. to load 2D pretrained models into 3D by repeating or something like that.

        # Now keep track of the # of layers with succesful weight transfers
        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(
            old_params.keys(), old_params.values(), new_params.values()
        ):
            # If more than one param in layer is NE (not equal) to the original weights we've successfully loaded new weights.
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
        """
        Factory method to create the appropriate model based on task type

        Args:
            task_type: Type of task (segmentation, classification, regression)
            config: Configuration dictionary
            **kwargs: Additional arguments for the model

        Returns:
            BaseSupervisedModel: Instance of appropriate model subclass
        """
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

from typing import Optional
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice
import numpy as np
from scipy import ndimage

from yucca.modules.optimization.loss_functions.deep_supervision import (
    DeepSupervisionLoss,
)
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE
from yucca.modules.metrics.training_metrics import F1

from models.supervised_base import BaseSupervisedModel

class FocalTversky(torch.nn.Module):

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        gt = (targets[:, 0] > 0).float()

        n = probs.numel()
        tp = (probs * gt).sum() / n
        fp = (probs * (1.0 - gt)).sum() / n
        fn = ((1.0 - probs) * gt).sum() / n

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky).pow(self.gamma)

class GeneralizedDiceLoss(torch.nn.Module):

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        gt = F.one_hot(targets[:, 0].long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        gt_vol = gt.sum(dim=dims)
        weights = 1.0 / (gt_vol.square() + self.smooth)

        intersect = (probs * gt).sum(dim=dims)
        denom = (probs + gt).sum(dim=dims)

        numerator = 2.0 * (weights * intersect).sum()
        denominator = (weights * denom).sum().clamp_min(self.smooth)
        return 1.0 - numerator / denominator

class TverskyCELoss(torch.nn.Module):

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        ce_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        gt = (targets[:, 0] > 0).float()

        tp = (probs * gt).sum()
        fp = (probs * (1.0 - gt)).sum()
        fn = ((1.0 - probs) * gt).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        tv_loss = 1.0 - tversky
        ce_loss = self.ce(logits, targets[:, 0].long())
        return tv_loss + self.ce_weight * ce_loss

class FocalTverskyCELoss(torch.nn.Module):

    def __init__(self, ce_weight: float = 0.5, fg_ce_weight: Optional[float] = None):
        super().__init__()
        self.ft = FocalTversky()
        self.ce_weight = ce_weight
        self.fg_ce_weight = fg_ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ft_loss = self.ft(logits, targets)
        if self.fg_ce_weight is None:
            ce_loss = F.cross_entropy(logits, targets[:, 0].long())
        else:
            weights = torch.tensor(
                [1.0, float(self.fg_ce_weight)],
                dtype=logits.dtype,
                device=logits.device,
            )
            ce_loss = F.cross_entropy(logits, targets[:, 0].long(), weight=weights)
        return ft_loss + self.ce_weight * ce_loss

class GeneralizedDiceCELoss(torch.nn.Module):

    def __init__(self, ce_weight: float = 0.5):
        super().__init__()
        self.gd = GeneralizedDiceLoss()
        self.ce_weight = ce_weight
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gd_loss = self.gd(logits, targets)
        ce_loss = self.ce(logits, targets[:, 0].long())
        return gd_loss + self.ce_weight * ce_loss

class ForegroundDiceCELoss(torch.nn.Module):

    def __init__(self, fg_ce_weight: Optional[float] = None):
        super().__init__()
        self.dice = DiceCE(
            soft_dice_kwargs={
                "apply_softmax": True,
                "do_bg": False,
                "batch_dice": True,
            },
            weight_ce=0,
            weight_dice=1,
        )
        self.fg_ce_weight = fg_ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        if self.fg_ce_weight is None:
            ce_loss = F.cross_entropy(logits, targets[:, 0].long())
        else:
            weights = torch.tensor(
                [1.0, float(self.fg_ce_weight)],
                dtype=logits.dtype,
                device=logits.device,
            )
            ce_loss = F.cross_entropy(logits, targets[:, 0].long(), weight=weights)
        return dice_loss + ce_loss

class SoftForegroundDiceLoss(torch.nn.Module):

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        gt = (targets[:, 0] > 0).float()
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * gt).sum(dim=dims)
        denominator = probs.sum(dim=dims) + gt.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()

class SizeAwareDiceCELoss(torch.nn.Module):

    def __init__(
        self,
        small_lesion_weight: float = 2.0,
        small_lesion_thresholds: tuple[int, int, int] = (1000, 10000, 50000),
    ):
        super().__init__()
        self.small_lesion_weight = float(small_lesion_weight)
        self.small_lesion_thresholds = tuple(int(v) for v in small_lesion_thresholds)
        self.dice = SoftForegroundDiceLoss()
        self._structure = np.ones((3, 3, 3), dtype=np.uint8)

    def _voxel_weights(self, targets: torch.Tensor) -> torch.Tensor:
        fg = (targets[:, 0] > 0).detach().cpu().numpy().astype(np.uint8, copy=False)
        weights = np.ones_like(fg, dtype=np.float32)
        if self.small_lesion_weight <= 0.0:
            return torch.from_numpy(weights).to(device=targets.device, dtype=torch.float32)

        t_small, t_medium, t_large = self.small_lesion_thresholds
        for sample_idx, sample_fg in enumerate(fg):
            if sample_fg.max() == 0:
                continue
            components, n_components = ndimage.label(sample_fg, structure=self._structure)
            if n_components == 0:
                continue
            component_sizes = np.bincount(components.ravel())[1:]
            sample_weights = weights[sample_idx]
            for label_idx, lesion_size in enumerate(component_sizes, start=1):
                if lesion_size <= t_small:
                    boost = self.small_lesion_weight
                elif lesion_size <= t_medium:
                    boost = 0.5 * self.small_lesion_weight
                elif lesion_size <= t_large:
                    boost = 0.25 * self.small_lesion_weight
                else:
                    boost = 0.0
                if boost > 0.0:
                    sample_weights[components == label_idx] = 1.0 + boost

        return torch.from_numpy(weights).to(device=targets.device, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        ce = F.cross_entropy(logits, targets[:, 0].long(), reduction="none")
        weights = self._voxel_weights(targets)
        ce_loss = (ce * weights).sum() / weights.sum().clamp_min(1.0)
        return dice_loss + ce_loss

class SupervisedSegModel(BaseSupervisedModel):

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
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=deep_supervision,
        )

    def _configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/dice": Dice(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                ),
                f"{prefix}/F1": F1(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                    average=None,
                ),
            },
        )

    def _configure_losses(self):
        loss_name = str(self.config.get("segmentation_loss") or "dicece").lower()
        if loss_name == "dicece":
            loss_fn_train = DiceCE(soft_dice_kwargs={"apply_softmax": True})
            loss_fn_val = DiceCE(soft_dice_kwargs={"apply_softmax": True})
        elif loss_name == "dicece_nobg":
            loss_fn_train = ForegroundDiceCELoss()
            loss_fn_val = ForegroundDiceCELoss()
        elif loss_name == "dicece_nobg_wce":
            loss_fn_train = ForegroundDiceCELoss(fg_ce_weight=20.0)
            loss_fn_val = ForegroundDiceCELoss(fg_ce_weight=20.0)
        elif loss_name == "focaltversky":
            loss_fn_train = FocalTversky()
            loss_fn_val = FocalTversky()
        elif loss_name == "focaltverskyce":
            loss_fn_train = FocalTverskyCELoss()
            loss_fn_val = FocalTverskyCELoss()
        elif loss_name == "focaltverskyce_wce":
            loss_fn_train = FocalTverskyCELoss(fg_ce_weight=20.0)
            loss_fn_val = FocalTverskyCELoss(fg_ce_weight=20.0)
        elif loss_name == "generalizeddice":
            loss_fn_train = GeneralizedDiceLoss()
            loss_fn_val = GeneralizedDiceLoss()
        elif loss_name == "generalizeddicece":
            loss_fn_train = GeneralizedDiceCELoss()
            loss_fn_val = GeneralizedDiceCELoss()
        elif loss_name == "tverskyce":
            loss_fn_train = TverskyCELoss()
            loss_fn_val = TverskyCELoss()
        elif loss_name == "dicece_sizeaware":
            thresholds = tuple(
                int(v)
                for v in self.config.get(
                    "seg_small_lesion_thresholds", [1000, 10000, 50000]
                )
            )
            weight = float(self.config.get("seg_small_lesion_weight") or 2.0)
            loss_fn_train = SizeAwareDiceCELoss(
                small_lesion_weight=weight,
                small_lesion_thresholds=thresholds,
            )
            loss_fn_val = SizeAwareDiceCELoss(
                small_lesion_weight=weight,
                small_lesion_thresholds=thresholds,
            )
        else:
            raise ValueError(f"Unsupported segmentation_loss: {loss_name}")

        if self.deep_supervision:
            loss_fn_train = DeepSupervisionLoss(loss_fn_train, weights=None)

        return loss_fn_train, loss_fn_val

    def _boundary_targets(self, target: torch.Tensor) -> torch.Tensor:
        fg = (target[:, 0] > 0).float().unsqueeze(1)
        radius = max(1, int(self.config.get("seg_boundary_radius") or 1))
        mode = str(self.config.get("seg_aux_target") or "binary_band").lower()
        boundary = fg.new_zeros(fg.shape)
        for step in range(1, radius + 1):
            kernel = 2 * step + 1
            dilated = F.max_pool3d(fg, kernel_size=kernel, stride=1, padding=step)
            eroded = -F.max_pool3d(-fg, kernel_size=kernel, stride=1, padding=step)
            band = (dilated - eroded).clamp_(0.0, 1.0)
            if mode == "distance_shell":
                weight = 1.0 - ((step - 1) / radius)
                boundary = torch.maximum(boundary, band * weight)
            else:
                boundary = torch.maximum(boundary, band)
        return boundary

    def _auxiliary_loss(self, target: torch.Tensor) -> torch.Tensor:
        aux_weight = float(self.config.get("seg_aux_loss_weight") or 0.0)
        if aux_weight <= 0.0:
            return target.new_zeros(())
        aux_outputs = getattr(self.model, "_last_aux_outputs", None)
        if not aux_outputs:
            return target.new_zeros(())
        boundary_logits = aux_outputs.get("boundary_logits")
        if boundary_logits is None:
            return target.new_zeros(())
        boundary_target = self._boundary_targets(target).to(boundary_logits.dtype)
        boundary_loss = F.binary_cross_entropy_with_logits(
            boundary_logits, boundary_target
        )
        return boundary_logits.new_tensor(aux_weight) * boundary_loss

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

        aux_loss = self._auxiliary_loss(target)
        total_loss = loss + aux_loss
        metrics = self.compute_metrics(self.train_metrics, output, target)
        to_log = {"train/loss": total_loss} | metrics
        if aux_loss.ndim == 0 and aux_loss.item() > 0:
            to_log["train/aux_loss"] = aux_loss
        self.log_dict(
            to_log,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_channels = batch["image"].shape[1]
        assert input_channels == self.num_modalities, (
            f"Expected {self.num_modalities} input channels, but got {input_channels}. "
            "This often happens when the task config has changed _after_ data has been preprocessed. "
            "Check your data preprocessing."
        )

        inputs, target, _ = self._process_batch(batch)
        output = self(inputs)
        loss = self.loss_fn_val(output, target)
        aux_loss = self._auxiliary_loss(target)
        total_loss = loss + aux_loss
        metrics = self.compute_metrics(self.val_metrics, output, target)
        to_log = {"val/loss": total_loss} | metrics
        if aux_loss.ndim == 0 and aux_loss.item() > 0:
            to_log["val/aux_loss"] = aux_loss
        self.log_dict(
            to_log,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def compute_metrics(self, metrics, output, target, ignore_index: int = 0):
        prefix = "seg"
        if metrics:
            first_key = next(iter(metrics.keys()))
            prefix = first_key.split("/", 1)[0] if "/" in first_key else prefix

        logged_metrics = metrics(output, target)
        tmp = {}
        to_drop = []
        for key in logged_metrics.keys():
            if logged_metrics[key].numel() > 1:
                to_drop.append(key)
                for i, val in enumerate(logged_metrics[key]):
                    if not i == ignore_index:
                        tmp[key + "_" + str(i)] = val
        for k in to_drop:
            logged_metrics.pop(k)

        if output.shape[1] > 1:
            probs = torch.softmax(output, dim=1)[:, 1]
            gt = (target[:, 0] > 0).float()
            dims = tuple(range(1, probs.ndim))
            intersection = (probs * gt).sum(dim=dims)
            denominator = probs.sum(dim=dims) + gt.sum(dim=dims)
            soft_dice = ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
            tmp[f"{prefix}/soft_dice"] = soft_dice

        logged_metrics.update(tmp)
        return logged_metrics

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef

from models.supervised_base import BaseSupervisedModel

class RegressionLoss(nn.Module):
    def __init__(self, beta=10.0):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss(beta=beta)

    def forward(self, logits, targets):
        return self.reg_loss(targets, logits)

class PairwiseRankLoss(nn.Module):

    def forward(self, logits, targets):
        pred = logits.reshape(-1)
        target = targets.reshape(-1)
        if pred.numel() < 2:
            return pred.new_zeros(())

        target_diff = target[:, None] - target[None, :]
        mask = torch.triu(target_diff.ne(0), diagonal=1)
        if not mask.any():
            return pred.new_zeros(())

        sign = torch.sign(target_diff)
        pred_diff = pred[:, None] - pred[None, :]
        losses = F.softplus(-(sign * pred_diff))
        return losses[mask].mean()

class CompositeRegressionLoss(nn.Module):
    def __init__(self, base_loss, rank_weight=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.rank_weight = float(rank_weight)
        self.rank_loss = PairwiseRankLoss()

    def forward(self, logits, targets):
        loss = self.base_loss(logits, targets)
        if self.rank_weight > 0:
            loss = loss + self.rank_weight * self.rank_loss(logits, targets)
        return loss

class SupervisedRegModel(BaseSupervisedModel):

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
            deep_supervision=False,
        )

    def _configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/mse": MeanSquaredError(),
                f"{prefix}/mae": MeanAbsoluteError(),
                f"{prefix}/pearson": PearsonCorrCoef(),
            }
        )

    def _configure_losses(self):
        loss_name = str(self.config.get("regression_loss") or "mse").lower()
        smoothl1_beta = float(self.config.get("regression_smoothl1_beta", 10.0))
        rank_weight = float(self.config.get("regression_rank_loss_weight", 0.0))

        if (
            self.model_name == "mmunetvae"
            and self.config.get("regression_loss") is None
        ):
            loss_name = "smoothl1"

        if loss_name == "smoothl1":
            base_loss = RegressionLoss(beta=smoothl1_beta)
        elif loss_name == "mse":
            base_loss = torch.nn.MSELoss()
        elif loss_name == "l1":
            base_loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsupported regression_loss: {loss_name}")
        loss_fn = CompositeRegressionLoss(base_loss, rank_weight=rank_weight)
        return loss_fn, loss_fn

    def _process_batch(self, batch):
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        target = target.float()
        return inputs, target, file_path

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        metrics.update(output, target)

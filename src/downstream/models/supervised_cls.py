from typing import Optional
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

from models.supervised_base import BaseSupervisedModel


class SupervisedClsModel(BaseSupervisedModel):
    """
    Supervised model for classification tasks.
    Inherits from BaseSupervisedModel and implements classification-specific functionality.
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
            deep_supervision=False,  # Classification doesn't use deep supervision
        )

    def _configure_metrics(self, prefix: str):
        """
        Configure classification-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of classification metrics
        """
        metrics_dict = {
            f"{prefix}/accuracy": Accuracy(
                task="multiclass", num_classes=self.num_classes
            ),
            f"{prefix}/precision": Precision(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ),
            f"{prefix}/recall": Recall(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ),
            f"{prefix}/f1": F1Score(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ),
            f"{prefix}/auroc": AUROC(
                task="multiclass",
                num_classes=self.num_classes,
            ),
        }
        return MetricCollection(metrics_dict)

    def _configure_losses(self):
        """
        Configure classification-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn, loss_fn

    def _process_batch(self, batch):
        """
        Process classification batch data

        Args:
            batch: Input batch

        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        target = target.long()

        if target.dim() > 1:
            target = target.squeeze(1)

        return inputs, target, file_path

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute classification metrics.
        All metrics (including AUROC) use multiclass mode and accept [N, C]
        probability input. This allows uniform MetricCollection.update().

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (not used in classification)
        """
        probabilities = F.softmax(output, dim=1)
        metrics.update(probabilities, target)

from typing import Any, Dict, Tuple
import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.loss import FocalLoss


class ECG_segmentation_LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, net: torch.nn.Module, learning_rate, focal_gamma, alpha, beta) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.learning_rate = learning_rate
        self.net = net
        self.alpha = alpha
        self.beta = beta
        # loss function
        self.loss_function_seg = FocalLoss(gamma=focal_gamma)
        self.loss_function_cls = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=4)
        self.seg_val_acc = Accuracy(task="multiclass", num_classes=4)
        self.cls_val_acc = Accuracy(task="multiclass", num_classes=2)
        # self.val_acc = Accuracy(task="multiclass", num_classes=4)
        self.seg_test_acc = Accuracy(task="multiclass", num_classes=4)
        self.cls_test_acc = Accuracy(task="multiclass", num_classes=2)
        # self.test_acc = Accuracy(task="multiclass", num_classes=4)

        # for averaging loss across batches
        self.seg_train_loss = MeanMetric()
        self.cls_train_loss = MeanMetric()
        self.train_loss = MeanMetric()

        self.seg_val_loss = MeanMetric()
        self.cls_val_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.seg_test_loss = MeanMetric()
        self.cls_test_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.seg_val_acc_best = MaxMetric()
        self.cls_val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.seg_val_loss.reset()
        self.cls_val_loss.reset()
        self.val_loss.reset()
        self.seg_val_acc.reset()
        self.cls_val_acc.reset()
        self.seg_val_acc_best.reset()
        self.cls_val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # x, y = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, y)
        # preds = torch.argmax(logits, dim=1)

        x, seg_targets, cls_targets = batch
        seg_preds, cls_probs_preds = self.forward(x)

        seg_loss = self.loss_function_seg(seg_preds, torch.argmax(seg_targets, dim=1))
        cls_loss = self.loss_function_cls(cls_probs_preds, cls_targets)
        loss = self.alpha * seg_loss + self.beta * cls_loss

        seg_preds = torch.argmax(seg_preds, dim=1)  # (B, 5000)
        cls_preds = torch.argmax(cls_probs_preds, dim=1)  # (B,)
        seg_targets = torch.argmax(seg_targets, dim=1)  # (B, 5000)

        return loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets

    def training_step( self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets = self.model_step(batch)

        # update and log metrics
        self.seg_train_loss(seg_loss)
        self.cls_train_loss(cls_loss)
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("seg_train/loss", self.seg_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_train/loss", self.cls_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets = self.model_step(batch)

        # update and log metrics
        self.seg_val_loss(seg_loss)
        self.cls_val_loss(cls_loss)
        self.val_loss(loss)

        self.seg_val_acc(seg_preds, seg_targets)
        self.cls_val_acc(cls_preds, cls_targets)

        self.log("seg_val/loss", self.seg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_val/loss", self.cls_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("seg_val/acc", self.seg_val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_val/acc", self.cls_val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        seg_acc = self.seg_val_acc.compute()  # get current val acc
        cls_acc = self.cls_val_acc.compute()
        self.seg_val_acc_best(seg_acc)  # update best so far val acc
        self.cls_val_acc_best(cls_acc)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("seg_val/acc_best", self.seg_val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("cls_val/acc_best", self.cls_val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_step_signals = []
        self.test_step_seg_preds = []
        self.test_step_cls_preds = []
        self.test_step_seg_targets = []
        self.test_step_cls_targets = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, _, _ = batch
        loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets = self.model_step(batch)

        # update and log metrics
        self.seg_test_loss(seg_loss)
        self.cls_test_loss(cls_loss)
        self.test_loss(loss)

        self.seg_test_acc(seg_preds, seg_targets)
        self.cls_test_acc(cls_preds, cls_targets)

        self.log("seg_test/loss", self.seg_test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_test/loss", self.cls_test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("seg_test/acc", self.seg_test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_test/acc", self.cls_test_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Lưu lại kết quả của batch này vào mảng
        self.test_step_signals.append(x.cpu())
        self.test_step_seg_preds.append(seg_preds.cpu())
        self.test_step_cls_preds.append(cls_preds.cpu())
        self.test_step_seg_targets.append(seg_targets.cpu())
        self.test_step_cls_targets.append(cls_targets.cpu())

    def on_test_epoch_end(self) -> None:
        # Ghép các batch lại thành mảng lớn
        signals = torch.cat(self.test_step_signals, dim=0).squeeze(1).numpy()
        all_seg_pred = torch.cat(self.test_step_seg_preds, dim=0).numpy()
        all_cls_pred = torch.cat(self.test_step_cls_preds, dim=0).numpy()
        seg_true = torch.cat(self.test_step_seg_targets, dim=0).numpy()
        cls_true = torch.cat(self.test_step_cls_targets, dim=0).numpy()

        # Lưu predictions.npz
        save_path = 'predictions.npz'
        np.savez(save_path,
                 seg_pred=all_seg_pred,
                 seg_true=seg_true,
                 cls_pred=all_cls_pred,
                 cls_true=cls_true,
                 signals=signals)

        # In kết quả chi tiết
        class_names = ['P', 'QRS', 'T', 'Baseline']
        print(f'\n{"=" * 50}')
        print(f'TEST RESULTS')
        print(f'{"=" * 50}')
        for c in range(4):
            p_c = (all_seg_pred == c)
            t_c = (seg_true == c)
            intersection = np.sum(p_c & t_c)
            union = np.sum(p_c) + np.sum(t_c)
            dice = 2.0 * intersection / union if union > 0 else 1.0
            print(f'Dice Score ({class_names[c]:>8s}): {dice:.4f}')
        print(f'{"=" * 50}')
        print(f'Predictions saved to: {save_path}')

        # Xóa mảng để giải phóng bộ nhớ
        self.test_step_signals.clear()
        self.test_step_seg_preds.clear()
        self.test_step_cls_preds.clear()
        self.test_step_seg_targets.clear()
        self.test_step_cls_targets.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.get("compile", False) and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learning_rate)
        if self.hparams.get("scheduler") is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


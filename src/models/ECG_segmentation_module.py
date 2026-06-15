from typing import Any, Dict, Tuple
import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.loss import FocalLoss

class ECG_segmentation_LitModule(LightningModule):
    def __init__(
        self, 
        net: torch.nn.Module, 
        learning_rate, 
        focal_gamma, 
        alpha, 
        beta, 
        scheduler: bool = True, 
        train_classification_only: bool = False, 
        pretrained_path: str = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.learning_rate = learning_rate
        self.net = net
        self.alpha = alpha
        self.beta = beta
        self.loss_function_seg = FocalLoss(gamma=focal_gamma)
        self.loss_function_cls = torch.nn.CrossEntropyLoss()

        self.seg_val_acc = Accuracy(task="multiclass", num_classes=4)
        self.cls_val_acc = Accuracy(task="multiclass", num_classes=2)
        self.seg_test_acc = Accuracy(task="multiclass", num_classes=4)
        self.cls_test_acc = Accuracy(task="multiclass", num_classes=2)

        self.seg_train_loss = MeanMetric()
        self.cls_train_loss = MeanMetric()
        self.train_loss = MeanMetric()

        self.seg_val_loss = MeanMetric()
        self.cls_val_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.seg_test_loss = MeanMetric()
        self.cls_test_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.seg_val_acc_best = MaxMetric()
        self.cls_val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.seg_val_loss.reset()
        self.cls_val_loss.reset()
        self.val_loss.reset()
        self.seg_val_acc.reset()
        self.cls_val_acc.reset()
        self.seg_val_acc_best.reset()
        self.cls_val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hparams.get('train_classification_only', False):
            x, _, cls_targets = batch  # DataModule luôn trả về 3 biến, bỏ qua dummy seg_targets
            seg_targets = None
        else:
            x, seg_targets, cls_targets = batch

        skip_decoder = self.hparams.get('train_classification_only', False)
        seg_preds, cls_probs_preds = self.net(x, skip_decoder=skip_decoder)

        cls_loss = self.loss_function_cls(cls_probs_preds, cls_targets)

        if self.hparams.get('train_classification_only', False):
            loss = cls_loss
            seg_loss = torch.tensor(0.0, device=self.device)
            seg_preds = torch.argmax(seg_preds, dim=1) if seg_preds is not None else None
            cls_preds = torch.argmax(cls_probs_preds, dim=1)
        else:
            seg_loss = self.loss_function_seg(seg_preds, torch.argmax(seg_targets, dim=1))
            loss = self.alpha * seg_loss + self.beta * cls_loss
            seg_preds = torch.argmax(seg_preds, dim=1)  # (B, 5000)
            cls_preds = torch.argmax(cls_probs_preds, dim=1)  # (B,)
            seg_targets = torch.argmax(seg_targets, dim=1)  # (B, 5000)

        return loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets = self.model_step(batch)

        if not self.hparams.get('train_classification_only', False):
            self.seg_train_loss(seg_loss)
            self.log("seg_train/loss", self.seg_train_loss, on_step=False, on_epoch=True, prog_bar=True)
            
        self.cls_train_loss(cls_loss)
        self.train_loss(loss)
        self.log("cls_train/loss", self.cls_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets = self.model_step(batch)

        if not self.hparams.get('train_classification_only', False):
            self.seg_val_loss(seg_loss)
            self.seg_val_acc(seg_preds, seg_targets)
            self.log("seg_val/loss", self.seg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("seg_val/acc", self.seg_val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.cls_val_loss(cls_loss)
        self.val_loss(loss)
        self.cls_val_acc(cls_preds, cls_targets)

        self.log("cls_val/loss", self.cls_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_val/acc", self.cls_val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if not self.hparams.get('train_classification_only', False):
            seg_acc = self.seg_val_acc.compute()
            self.seg_val_acc_best(seg_acc)
            self.log("seg_val/acc_best", self.seg_val_acc_best.compute(), sync_dist=True, prog_bar=True)

        cls_acc = self.cls_val_acc.compute()
        self.cls_val_acc_best(cls_acc)
        self.log("cls_val/acc_best", self.cls_val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_step_signals = []
        self.test_step_seg_preds = []
        self.test_step_cls_preds = []
        self.test_step_seg_targets = []
        self.test_step_cls_targets = []

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        if self.hparams.get('train_classification_only', False):
            x, _ = batch
        else:
            x, _, _ = batch
            
        loss, seg_loss, cls_loss, seg_preds, cls_preds, seg_targets, cls_targets = self.model_step(batch)

        if not self.hparams.get('train_classification_only', False):
            self.seg_test_loss(seg_loss)
            self.seg_test_acc(seg_preds, seg_targets)
            self.log("seg_test/loss", self.seg_test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("seg_test/acc", self.seg_test_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.test_step_seg_targets.append(seg_targets.cpu())

        self.cls_test_loss(cls_loss)
        self.test_loss(loss)
        self.cls_test_acc(cls_preds, cls_targets)

        self.log("cls_test/loss", self.cls_test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_test/acc", self.cls_test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_step_signals.append(x.cpu())
        self.test_step_seg_preds.append(seg_preds.cpu())
        self.test_step_cls_preds.append(cls_preds.cpu())
        self.test_step_cls_targets.append(cls_targets.cpu())

    def on_test_epoch_end(self) -> None:
        signals = torch.cat(self.test_step_signals, dim=0).squeeze(1).numpy()
        all_seg_pred = torch.cat(self.test_step_seg_preds, dim=0).numpy()
        all_cls_pred = torch.cat(self.test_step_cls_preds, dim=0).numpy()
        cls_true = torch.cat(self.test_step_cls_targets, dim=0).numpy()

        save_dict = {
            'seg_pred': all_seg_pred,
            'cls_pred': all_cls_pred,
            'cls_true': cls_true,
            'signals': signals
        }

        if not self.hparams.get('train_classification_only', False):
            seg_true = torch.cat(self.test_step_seg_targets, dim=0).numpy()
            save_dict['seg_true'] = seg_true
            
        save_path = 'predictions.npz'
        np.savez(save_path, **save_dict)

        if not self.hparams.get('train_classification_only', False):
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

        self.test_step_signals.clear()
        self.test_step_seg_preds.clear()
        self.test_step_cls_preds.clear()
        self.test_step_cls_targets.clear()
        if not self.hparams.get('train_classification_only', False):
            self.test_step_seg_targets.clear()

    def setup(self, stage: str) -> None:
        if self.hparams.get("compile", False) and stage == "fit":
            self.net = torch.compile(self.net)
            
        pretrained_path = self.hparams.get("pretrained_path")
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = {k.replace("net.", ""): v for k, v in checkpoint["state_dict"].items() if not k.startswith("loss_function")}
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            self.net.load_state_dict(state_dict, strict=False)
            
        if self.hparams.get("train_classification_only", False):
            print("train_classification_only = True -> Freezing non-classify parameters...")
            for name, param in self.net.named_parameters():
                if "classify" not in name:
                    param.requires_grad = False

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(params=trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "cls_val/loss" if self.hparams.get("train_classification_only", False) else "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

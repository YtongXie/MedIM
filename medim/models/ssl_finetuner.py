from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy, F1Score
# from torchmetrics.classification import MultilabelAccuracy

class SSLFineTuner(LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 model_name: str = "vit_base",
                 dataset: str = "vindr",
                 in_features: int = 2048,
                 num_classes: int = 5,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 learning_rate: float = 5e-4,  #
                 weight_decay: float = 1e-6,
                 multilabel: bool = True,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dataset = dataset

        if multilabel:
            self.train_auc = AUROC(num_classes=num_classes)
            self.val_auc = AUROC(num_classes=num_classes,
                                 compute_on_step=False)
            self.test_auc = AUROC(num_classes=num_classes,
                                  compute_on_step=False)
            self.test_acc = Accuracy(
                num_classes=num_classes, compute_on_step=False, average='macro')
            self.test_f1 = F1Score(task="multilabel", num_classes=num_classes, average='macro')

        else:
            self.train_acc = Accuracy(num_classes=num_classes, topk=1)
            self.val_acc = Accuracy(
                num_classes=num_classes, topk=1, compute_on_step=False)
            self.test_acc = Accuracy(
                num_classes=num_classes, topk=1, compute_on_step=False)
            self.test_auc = AUROC(task='multiclass', num_classes=num_classes, compute_on_step=False)
            self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.save_hyperparameters(ignore=['backbone'])

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.model_name = model_name
        self.multilabel = multilabel

        self.linear_layer = SSLEvaluator(
            n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            auc = self.train_auc(torch.sigmoid(logits).float(), y.long())
            self.log("train_auc_step", auc, prog_bar=True, sync_dist=True)
            self.log("train_auc_epoch", self.train_auc,
                     prog_bar=True, sync_dist=True)
        else:
            acc = self.train_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("train_acc_step", acc, prog_bar=True, sync_dist=True)
            self.log("train_acc_epoch", self.train_acc,
                     prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            self.val_auc(torch.sigmoid(logits).float(), y.long())
            self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.val_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        if self.multilabel:
            self.test_auc(torch.sigmoid(logits).float(), y.long())
            print(self.test_auc(torch.sigmoid(logits).float(), y.long()))
            self.log("test_auc", self.test_auc, on_epoch=True)

            print(self.test_acc(torch.sigmoid(logits).float(), y.long()))
            self.log("test_acc", self.test_acc, on_epoch=True)

            print(self.test_f1(torch.sigmoid(logits).float(), y.long()))
            self.log("test_f1", self.test_f1, on_epoch=True)

        else:
            self.test_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("test_acc", self.test_acc, on_epoch=True)

            self.test_auc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("test_auc", self.test_auc, on_epoch=True)

            self.test_f1(F.softmax(logits, dim=-1).float(), y.long())
            self.log("test_f1", self.test_f1, on_epoch=True)

        return loss

    def shared_step(self, batch, grad=True):
        x, y = batch

        if len(batch[0].size()) > 4:
            bs, n_crops, c, h, w =  x.size() 
            x = x.view(-1, c, h, w)
            # For multi-class
            with torch.no_grad():
                feats, _ = self.backbone(x)
                logits = self.linear_layer(feats.view(feats.size(0), -1))

            if self.multilabel:
                loss =F.binary_cross_entropy_with_logits(logits.view(bs, n_crops, -1).mean(1).float(), y.float())
                logits = logits.view(bs, n_crops, -1).mean(1)

            else:
                y = y.squeeze()
                loss = F.cross_entropy(logits.view(bs, n_crops, -1).mean(1).float(), y.long())
                logits = F.softmax(logits, dim=-1).float()
                logits = logits.view(bs, n_crops, -1).mean(1)
        else:
            # For multi-class
            if grad:
                feats, _ = self.backbone(x)
            else:
                with torch.no_grad():
                    feats, _ = self.backbone(x)

            feats = feats.view(feats.size(0), -1)
            logits = self.linear_layer(feats)

            if self.multilabel:
                if self.dataset == 'vindr':
                    weights = [ 0.38108194,  6.13937848,  2.54842126,  0.50913532, 45.02210884,
                                3.26022167, 94.54642857, 14.32521645,  9.74705449,  3.00147392,
                                1.92951895,  0.89110677, 21.48782468, 36.36401099,  7.81375443,
                                1.38834697,  1.15019986,  0.58907432, 13.13144841,  0.71953142,
                                12.7765444 ,  1.00581307, 32.60221675,  4.09291899,  1.29161788,
                                1.6106717 ,  0.26951662,  0.11157237]
                    loss_func = nn.MultiLabelSoftMarginLoss(weight=torch.FloatTensor(weights).to(logits.device))
                    loss = loss_func(logits.float(), y.float())

                else:
                    loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())


                logits = torch.sigmoid(logits).float()
            else:
                y = y.squeeze()
                loss = F.cross_entropy(logits.float(), y.long())
                logits = F.softmax(logits, dim=-1).float()

        return loss, logits, y


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden), 
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

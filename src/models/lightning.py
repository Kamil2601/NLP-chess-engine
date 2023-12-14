import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch


class LitHuggingfaceClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5, weight_decay=0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.valid_acc_micro = Accuracy(task="multiclass", average='micro', num_classes=2)
        self.valid_acc_macro = Accuracy(task="multiclass", average='macro', num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs.logits
        loss = outputs.loss

        preds = torch.argmax(logits, dim=-1)

        y = batch["labels"]
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()
    
    def on_validation_epoch_end(self) -> None:
        self.log("valid_acc_micro", self.valid_acc_micro.compute(), prog_bar=True)
        self.log("valid_acc_macro", self.valid_acc_macro.compute(), prog_bar=True)
        self.valid_acc_macro.reset()
        self.valid_acc_micro.reset()


    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs.logits
        loss = outputs.loss

        # preds = torch.argmax(logits, dim=-1)

        y = batch["labels"]
        self.valid_acc_micro.update(logits, y)
        self.valid_acc_macro.update(logits, y)
        # acc_micro = self.valid_acc_micro(preds, y)
        # acc_macro = self.valid_acc_macro(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        # self.log("valid_acc_micro", acc_micro, prog_bar=True)
        # self.log("valid_acc_macro", acc_macro, prog_bar=True)
        # self.log("valid_acc_micro", self.valid_acc_micro.compute(), prog_bar=True)
        # self.log("valid_acc_macro", self.valid_acc_macro.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
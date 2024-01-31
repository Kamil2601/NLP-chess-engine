from typing import Any
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from pathlib import Path
import torch.nn.functional as F

class LitHuggingfaceClassifier(pl.LightningModule):
    def __init__(self, checkpoint, learning_rate=2e-5, weight_decay=0.0, save_dir = None):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.valid_acc_micro = Accuracy(task="multiclass", average='micro', num_classes=2)
        self.valid_acc_macro = Accuracy(task="multiclass", average='macro', num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.save_dir = save_dir

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
        if self.save_dir:
            self.model.save_pretrained(f"{self.save_dir}/epoch_{self.current_epoch}")

    
    def on_validation_epoch_end(self) -> None:
        self.log("valid_acc_micro", self.valid_acc_micro.compute(), prog_bar=True)
        self.log("valid_acc_macro", self.valid_acc_macro.compute(), prog_bar=True)
        self.valid_acc_macro.reset()
        self.valid_acc_micro.reset()


    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs.logits
        loss = outputs.loss

        y = batch["labels"]
        self.valid_acc_micro.update(logits, y)
        self.valid_acc_macro.update(logits, y)
        self.log("valid_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx) -> Any:
        outputs = self.model(**batch)
        logits = outputs.logits
        return torch.argmax(logits, dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    

class PLClassifierModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.valid_acc_micro = Accuracy(task="multiclass", average='micro', num_classes=2)
        self.valid_acc_macro = Accuracy(task="multiclass", average='macro', num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)


    def forward(self, x):
        return self.model(x)
    
    def _loss_function(self, logits, labels):
        if logits.shape[-1] == 1:
            return F.binary_cross_entropy_with_logits(logits, labels)
        else:
            return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["input"], batch["label"]
        logits = self(inputs)

        preds = torch.argmax(logits, dim=-1)

        loss = self._loss_function(logits, labels)

        self.train_acc.update(preds, labels)
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
        inputs, labels = batch["input"], batch["label"]
        logits = self(inputs)

        preds = torch.argmax(logits, dim=-1)
        loss = self._loss_function(logits, labels)

        self.valid_acc_micro.update(preds, labels)
        self.valid_acc_macro.update(preds, labels)
        self.log("valid_loss", loss, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
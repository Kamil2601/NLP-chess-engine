import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, Dataset
from .samplers import BySequenceLengthBatchSampler2
from IPython.display import clear_output
import torch


class HuggingFaceTokenizedDataModule(pl.LightningDataModule):
    def __init__(self, checkpoint, batch_size = 8, dataset_path = "~/Projects/Master-Thesis/data/datasets/comments_with_color/"):
        super().__init__()
        self.checkpoint = checkpoint
        self.sampler = None
        self.dataset_path = dataset_path
        self.eval_batch_size = batch_size
        

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.raw_datasets = load_from_disk(self.dataset_path)

        tokenize_function = lambda example: self.tokenizer(example["color_comment"], truncation=True)

        self.tokenized_datasets = self.raw_datasets.map(tokenize_function, batched=True)

        clear_output(wait=True)

        self.tokenized_datasets = self.tokenized_datasets.remove_columns(["fen", "move", "comment", "color_comment"])
        self.tokenized_datasets = self.tokenized_datasets.rename_column("sentiment", "labels")
        self.tokenized_datasets.set_format("torch")

        # print(self.tokenized_datasets)


        self.train_dataset = self.tokenized_datasets["train"]
        self.val_dataset = self.tokenized_datasets["validation"]
        self.test_dataset = self.tokenized_datasets["test"]

        self.sampler = BySequenceLengthBatchSampler2(self.train_dataset["input_ids"])

    def train_dataloader(self):
        if self.sampler:
            return DataLoader(self.train_dataset, batch_sampler=self.sampler, collate_fn=self.data_collator)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.eval_batch_size, collate_fn=self.data_collator)
    

class MoveEvaluationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_path = "~/Projects/Master-Thesis/data/huggingface/my_datasets/sentimate_v1_with_tensors/") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path

    def setup(self, stage=None):
        dataset = Dataset.load_from_disk()
        dataset.set_format(type='torch', columns=["tensor", "sentiment"], dtype=torch.float32)
        dataset = dataset.rename_column("sentiment", "label")
        dataset = dataset.rename_column("tensor", "input")
        split = dataset.train_test_split(test_size=0.005)
        self.train_dataset = split["train"]
        self.val_dataset = split["test"]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


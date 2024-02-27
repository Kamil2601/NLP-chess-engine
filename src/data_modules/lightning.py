import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, Dataset
from .samplers import BySequenceLengthBatchSampler2
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import torch
from .move_as_tensor import MoveAsTensorDatasetLazy
import board_representation.sentimate as br_sentimate


class HuggingFaceTokenizedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        checkpoint,
        batch_size=8,
        dataset_path="~/Projects/Master-Thesis/data/datasets/comments_with_color/",
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.sampler = None
        self.dataset_path = dataset_path
        self.eval_batch_size = batch_size

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.raw_datasets = load_from_disk(self.dataset_path)

        tokenize_function = lambda example: self.tokenizer(
            example["color_comment"], truncation=True
        )

        self.tokenized_datasets = self.raw_datasets.map(tokenize_function, batched=True)

        clear_output(wait=True)

        self.tokenized_datasets = self.tokenized_datasets.remove_columns(
            ["fen", "move", "comment", "color_comment"]
        )
        self.tokenized_datasets = self.tokenized_datasets.rename_column(
            "sentiment", "labels"
        )
        self.tokenized_datasets.set_format("torch")

        # print(self.tokenized_datasets)

        self.train_dataset = self.tokenized_datasets["train"]
        self.val_dataset = self.tokenized_datasets["validation"]
        self.test_dataset = self.tokenized_datasets["test"]

        self.sampler = BySequenceLengthBatchSampler2(self.train_dataset["input_ids"])

    def train_dataloader(self):
        if self.sampler:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.sampler,
                collate_fn=self.data_collator,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.eval_batch_size, collate_fn=self.data_collator
        )


class MoveEvaluationDataModuleHFDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=128,
        num_workers=8,
        dataset_path="~/Projects/Master-Thesis/data/huggingface/my_datasets/sentimate_v1_with_tensors/",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.num_workers = num_workers

    def setup(self, stage=None):
        # load Huggingface dataset from disk
        dataset = Dataset.load_from_disk(self.dataset_path)
        dataset.set_format(
            type="torch", columns=["tensor", "sentiment"], dtype=torch.float32
        )
        dataset.set_format(
            type="torch",
            columns=["sentiment"],
            dtype=torch.int32,
            output_all_columns=True,
        )
        dataset.set_format(type="torch", columns=["tensor", "sentiment"])

        # label - tensor(0) or tensor(1)
        # input - tensor of shape 26x8x8
        dataset = dataset.rename_column("sentiment", "label")
        dataset = dataset.rename_column("tensor", "input")
        split = dataset.train_test_split(test_size=0.005)
        self.train_dataset = split["train"]
        self.val_dataset = split["test"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MoveEvaluationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        validation_size=0.01,
        dataframe: pd.DataFrame | None = None,
        train_dataset=None,
        val_dataset=None,
        num_workers=2,
        piece_list_to_array_fn=br_sentimate.piece_lists_to_board_array_only_pieces,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.train_df = None
        self.val_df = None
        self.val_dataset = train_dataset
        self.train_dataset = val_dataset
        self.num_workers = num_workers
        self.piece_list_to_array_fn = piece_list_to_array_fn

    def setup(self, stage=None):
        if self.train_df is None:
            self.train_df, self.val_df = train_test_split(
                self.dataframe, test_size=self.validation_size
            )
            print(
                f"Train size: {len(self.train_df)}, Validation size: {len(self.val_df)}"
            )

        if self.train_dataset is None and self.train_df is not None:
            self.train_dataset = MoveAsTensorDatasetLazy(self.train_df, piece_list_to_array_fn=self.piece_list_to_array_fn)

        if self.val_dataset is None and self.val_df is not None:
            self.val_dataset = MoveAsTensorDatasetLazy(self.val_df, piece_list_to_array_fn=self.piece_list_to_array_fn)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

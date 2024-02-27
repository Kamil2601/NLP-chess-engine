from torch.utils.data import Dataset
import torch
from tqdm import tqdm

import board_representation.sentimate as br_sentimate
import pandas as pd
import multiprocessing as mp
import pyarrow as pa
import numpy as np
from joblib import delayed, Parallel


# def _move_to_tensor_tuple(move):
#     return move_to_tensor(move[0], move[1])


class MoveAsTensorDataset(Dataset):
    def __init__(
        self,
        moves_df: pd.DataFrame,
        position_col="position",
        move_col="move",
        sentiment_col="sentiment",
        convert_fn=br_sentimate.move_to_tensor,
    ):
        self.sentiments = [
            torch.tensor([s], dtype=torch.float16) for s in moves_df[sentiment_col]
        ]
        self.moves_df = moves_df
        moves_tuples = list(zip(moves_df[position_col], moves_df[move_col]))
        # self.moves = list(map(_move_to_tensor_tuple, moves_tuples))
        self.moves = list(map(lambda move: convert_fn(move[0], move[1]), moves_tuples))

    def __getitem__(self, index):
        return self.moves[index], self.sentiments[index]

    def __len__(self):
        return len(self.sentiments)


class MoveAsTensorDatasetOnlineTransform(Dataset):
    def __init__(
        self,
        moves_df: pd.DataFrame,
        position_col="position",
        move_col="move",
        sentiment_col="sentiment",
        convert_fn=br_sentimate.move_to_tensor,
    ):
        self.sentiments = [
            torch.tensor([s], dtype=torch.float16) for s in moves_df[sentiment_col]
        ]
        self.moves_df = moves_df
        self.moves_tuples = list(zip(moves_df[position_col], moves_df[move_col]))
        self.convert_fn = convert_fn

    def __getitem__(self, index):
        return (
            self.convert_fn(self.moves_tuples[index][0], self.moves_tuples[index][1]),
            self.sentiments[index],
        )

    def __len__(self):
        return len(self.sentiments)


class MoveAsTensorDatasetLazy(Dataset):
    def __init__(
        self,
        moves_df: pd.DataFrame,
        position_col="position",
        move_col="move",
        sentiment_col="sentiment",
        piece_list_to_array_fn=br_sentimate.piece_lists_to_board_array_only_pieces,
        ):
        # self.sentiments = [torch.tensor(s, dtype=torch.long) for s in moves_df[sentiment_col]]
        self.sentiments = torch.tensor(list(moves_df[sentiment_col]), dtype=torch.long)
        self.moves_df = moves_df
        self.piece_list_to_array_fn = piece_list_to_array_fn

        moves_tuples = list(zip(moves_df[position_col], moves_df[move_col]))
        self.pieces_lists = list(
            map(lambda x: br_sentimate.move_to_piece_list(x[0], x[1]), moves_tuples)
        )
        self.pieces_lists = pa.array(self.pieces_lists)

    def __getitem__(self, index):
        piece_lists = self.pieces_lists[index]
        piece_lists = [
            np.array(piece_list.as_py(), dtype=np.int16) for piece_list in piece_lists
        ]
        move_array = self.piece_list_to_array_fn(*piece_lists)
        move_tensor = torch.from_numpy(move_array).to(dtype=torch.float32)
        return move_tensor, self.sentiments[index]

    def __len__(self):
        return len(self.sentiments)

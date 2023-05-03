from torch.utils.data import Dataset
import torch
from board_representation import move_to_tensor
import pandas as pd
import multiprocessing as mp


def _move_to_tensor_tuple(move):
    return move_to_tensor(move[0], move[1])


class MoveAsTensorDataset(Dataset):
    def __init__(self, moves_df: pd.DataFrame, position_col="position", move_col="move", sentiment_col="sentiment"):
        self.sentiments = [torch.tensor([s], dtype=torch.float16) for s in moves_df[sentiment_col]]
        self.moves_df = moves_df
        moves_tuples = list(zip(moves_df[position_col], moves_df[move_col]))
        self.moves = list(map(_move_to_tensor_tuple, moves_tuples))

    def __getitem__(self, index):
        return self.moves[index], self.sentiments[index]

    def __len__(self):
        return len(self.sentiments)

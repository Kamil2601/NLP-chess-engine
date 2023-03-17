from torch.utils.data import Dataset
import torch
from board_representation import move_to_tensor
import pandas as pd


class MoveAsTensorDataset(Dataset):
    def __init__(self, moves_df: pd.DataFrame, position_col="position", move_col="move", sentiment_col="sentiment"):
        self.sentiments = [torch.tensor([s], dtype=torch.float32) for s in moves_df[sentiment_col]]
        self.moves = [move_to_tensor(row[position_col], row[move_col]) for _, row in moves_df.iterrows()]

    def __getitem__(self, index):
        return self.moves[index], self.sentiments[index]

    def __len__(self):
        return len(self.sentiments)

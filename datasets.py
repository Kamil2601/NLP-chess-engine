import torch
from torch.utils.data import Dataset
import torchtext
import random
import sqlite3 as db
import pandas as pd

def load_sql_to_df(sql, db_filename):
    con = db.connect(db_filename)
    data_frame = pd.read_sql_query(sql, con)

    return data_frame


def add_padding_vector_to_embeddings(embeddings: torchtext.vocab.Vectors, padding_token="<PAD>"):
    embeddings.stoi = {token: index + 1 for token, index in embeddings.stoi.items()}
    embeddings.stoi[padding_token] = 0
    embeddings.itos = [padding_token] + embeddings.itos

    padding_vector = torch.zeros((1, embeddings.dim), dtype=embeddings.vectors.dtype)

    embeddings.vectors = torch.cat((padding_vector, embeddings.vectors))



class PretrainedEmbeddingsDataset(Dataset):
    def __init__(self, data, embeddings: torchtext.vocab.Vectors):
        """
            data: [(tokens, sentiment)]
            embeddings: pretrained torchtext embeddings e.g. torchtext.vocab.GloVe
        """
        _data = data[:]
        random.shuffle(_data)
        self.data = sorted(_data, key=lambda item: len(item[0]))
        self.embeddings = embeddings

    def __getitem__(self, index):
        tokens, sentiment = self.data[index]

        return self.embeddings.get_vecs_by_tokens(tokens), torch.tensor(sentiment)

    def __len__(self):
        return len(self.data)



class PretrainedEmbeddingsIndicesDataset(Dataset):
    def __init__(self, data, embeddings: torchtext.vocab.Vectors):
        """
            data: [(tokens, sentiment)]
            embeddings: pretrained torchtext embeddings e.g. torchtext.vocab.GloVe
        """
        _data = data[:]
        random.shuffle(_data)
        self.data = sorted(_data, key=lambda item: len(item[0]))
        self.embeddings = embeddings

    def __getitem__(self, index):
        tokens, sentiment = self.data[index]

        indices = [self.embeddings.stoi[t] for t in tokens]

        return torch.tensor(indices), torch.tensor(sentiment)

    def __len__(self):
        return len(self.data)
import torch
from torch.utils.data import Dataset
import torchtext
import random
import sqlite3 as db
import pandas as pd
import torch.nn as nn

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


def DataLoaderPadding(**kwargs):
    def collate_batch(data):
        inputs = [item[0] for item in data]
        targets = [item[1] for item in data]

        input_batch = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        target_batch = torch.stack(targets)

        return input_batch, target_batch
    
    return torch.utils.data.DataLoader(**kwargs, collate_fn=collate_batch)



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
    def __init__(self, moves_df: pd.DataFrame, embeddings: torchtext.vocab.Vectors, comment_col="comment", sentiment_col="sentiment"):
        commments_ind = [torch.tensor([embeddings.stoi[t] for t in com], dtype=torch.int32) for com in moves_df[comment_col]]

        if sentiment_col != None:
            sentiments = [torch.tensor([sent], dtype=torch.float32) for sent in moves_df[sentiment_col]]            
            self.data = list(zip(commments_ind, sentiments))
        else:
            self.data = [(com,) for com in commments_ind]

        self.data.sort(key = lambda item: len(item[0]))

        self.embeddings = embeddings

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
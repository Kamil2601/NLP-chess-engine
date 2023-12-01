import torch
from torch.utils.data import Dataset
import torchtext
import random
import sqlite3 as db
import pandas as pd
import torch.nn as nn
# from torch.data import Sampler



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



# class BySequenceLengthSampler(Sampler):

#     def __init__(self, data_source,  
#                 bucket_boundaries, batch_size=64,):
#         ind_n_len = []
#         for i, p in enumerate(data_source):
#             ind_n_len.append( (i, p.shape[0]) )
#         self.ind_n_len = ind_n_len
#         self.bucket_boundaries = bucket_boundaries
#         self.batch_size = batch_size
        
        
#     def __iter__(self):
#         data_buckets = dict()
#         # where p is the id number and seq_len is the length of this id number. 
#         for p, seq_len in self.ind_n_len:
#             pid = self.element_to_bucket_id(p,seq_len)
#             if pid in data_buckets.keys():
#                 data_buckets[pid].append(p)
#             else:
#                 data_buckets[pid] = [p]

#         for k in data_buckets.keys():

#             data_buckets[k] = np.asarray(data_buckets[k])

#         iter_list = []
#         for k in data_buckets.keys():
#             np.random.shuffle(data_buckets[k])
#             iter_list += (np.array_split(data_buckets[k]
#                            , int(data_buckets[k].shape[0]/self.batch_size)))
#         shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
#         # size
#         for i in iter_list: 
#             yield i.tolist() # as it was stored in an array
    
#     def __len__(self):
#         return len(self.data_source)
    
#     def element_to_bucket_id(self, x, seq_length):
#         boundaries = list(self.bucket_boundaries)
#         buckets_min = [np.iinfo(np.int32).min] + boundaries
#         buckets_max = boundaries + [np.iinfo(np.int32).max]
#         conditions_c = np.logical_and(
#           np.less_equal(buckets_min, seq_length),
#           np.less(seq_length, buckets_max))
#         bucket_id = np.min(np.where(conditions_c))
#         return bucket_id
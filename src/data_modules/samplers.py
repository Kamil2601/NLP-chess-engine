from random import shuffle

import numpy as np


class BySequenceLengthBatchSampler:
    def __init__(self, data_source, batch_size=32) -> None:
        self.ind_n_len = [(i, len(sequence)) for i, sequence in enumerate(data_source)]
        shuffle(self.ind_n_len)
        self.ind_n_len.sort(key=lambda x: x[1])
        self.ind_n_len = np.array(self.ind_n_len)
        self.batches = np.array_split(
            self.ind_n_len[:, 0], len(self.ind_n_len) // batch_size
        )

    def __iter__(self):
        shuffle(self.batches)
        for bucket in self.batches:
            yield bucket.tolist()

    def __len__(self):
        return len(self.batches)
    
def batches_max_length(ind_n_len, start_index = 0, batch_size=32, max_length=128):
    batches = []

    for i in range(start_index, len(ind_n_len), batch_size):
        batch = ind_n_len[i:i+batch_size]
        if len(batch) > 0 and batch[-1, 1] <= max_length:
            batches.append(batch[:,0])
        else:
            break

    return batches, start_index + len(batches) * batch_size


class BySequenceLengthBatchSampler2:
    def __init__(self, data_source, batch_size=[32, 16, 8, 4], seq_lenghts = [64, 128, 256]) -> None:
        self.ind_n_len = [(i, len(sequence)) for i, sequence in enumerate(data_source)]
        shuffle(self.ind_n_len)
        self.ind_n_len.sort(key=lambda x: x[1])
        self.ind_n_len = np.array(self.ind_n_len)
        self.batch_size = batch_size
        self.seq_lenghts = seq_lenghts

        self.batches = []

        seq_lenghts.append(np.inf)

        i = 0

        for bs, sl in zip(batch_size, seq_lenghts):
            batches, i = batches_max_length(self.ind_n_len, i, bs, sl)
            self.batches.extend(batches)


    def __iter__(self):
        shuffle(self.batches)
        for batch in self.batches:
            yield batch.tolist()

    def __len__(self):
        return len(self.batches)
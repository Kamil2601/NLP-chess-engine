import torch.nn as nn
import torchtext
import torch


def _create_embedding_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


class SentimentAnalysisLSTM(nn.Module):
    def __init__(
        self,
        embeddings: torchtext.vocab.Vectors,
        hidden_dim,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding = _create_embedding_layer(embeddings.vectors, non_trainable=True)

        self.lstm = nn.LSTM(
            input_size=embeddings.dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)
        last_output = output[:, -1, :]

        hidden = (
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            if self.lstm.bidirectional
            else hidden[-1, :, :]
        )

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits


class SentimentAnalysisRNN(nn.Module):
    def __init__(
        self,
        embeddings: torchtext.vocab.Vectors,
        hidden_dim,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding = _create_embedding_layer(embeddings.vectors, non_trainable=True)

        self.lstm = nn.RNN(
            input_size=embeddings.dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.lstm(embedded)
        last_output = output[:, -1, :]

        hidden = (
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            if self.lstm.bidirectional
            else hidden[-1, :, :]
        )

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits


class SentimentAnalysisGRU(nn.Module):
    def __init__(
        self,
        embeddings: torchtext.vocab.Vectors,
        hidden_dim,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding = _create_embedding_layer(embeddings.vectors, non_trainable=True)

        self.lstm = nn.GRU(
            input_size=embeddings.dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.lstm(embedded)
        last_output = output[:, -1, :]

        hidden = (
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            if self.lstm.bidirectional
            else hidden[-1, :, :]
        )

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits

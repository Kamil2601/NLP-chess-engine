import torch.nn as nn
import torchtext

def _create_embedding_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer

class SentimentAnalysisLSTM(nn.Module):
    def __init__(self, embeddings: torchtext.vocab.Vectors, hidden_dim, num_layers = 2, dropout = 0.2):
        super().__init__()
        
        self.embedding = _create_embedding_layer(embeddings.vectors, non_trainable=True)

        self.lstm = nn.LSTM(input_size=embeddings.dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)
        
        
    def forward(self, x):
        embedded = self.embedding(x)

        lstm_output, _ = self.lstm(embedded)
        last_lstm_output = lstm_output[:, -1, :]

        logits = self.fc(last_lstm_output)

        return logits
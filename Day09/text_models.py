import torch
import torch.nn as nn


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        summed = embedded.sum(dim=1)
        lengths = lengths.to(token_ids.device).unsqueeze(1).clamp_min(1)
        mean_embedding = summed / lengths
        return self.classifier(mean_embedding)


class RNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        hidden_size,
        num_classes,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        output, _ = self.rnn(embedded)
        batch_indices = torch.arange(
            token_ids.size(0),
            device=token_ids.device,
        )
        last_valid_indices = lengths.to(token_ids.device) - 1
        last_hidden = output[batch_indices, last_valid_indices]
        return self.classifier(last_hidden)

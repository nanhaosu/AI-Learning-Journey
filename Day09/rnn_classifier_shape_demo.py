import torch
import torch.nn as nn

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
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self,token_ids,lengths):

        embedded = self.embedding(token_ids)

        output , h_n = self.rnn(embedded)

        batch_indices = torch.arange(
            token_ids.size(0),
            device=token_ids.device,
        )

        last_valid_indices = lengths.to(token_ids.device) - 1

        last_hidden = output[
            batch_indices,
            last_valid_indices,
        ]

        logits =  self.classifier(last_hidden)

        return logits


token_ids = torch.tensor(
    [
        [2, 3, 4, 5],
        [6, 7, 0, 0],
        [8, 9, 10, 0],
    ]
)

lengths = torch.tensor([4, 2, 3])

model = RNNTextClassifier(
    vocabulary_size=20,
    embedding_dim=8,
    hidden_size=16,
    num_classes=2,
)

logits = model(token_ids, lengths)

print("Token IDs:", token_ids.shape)
print("Lengths:  ", lengths.shape)
print("Logits:   ", logits.shape)
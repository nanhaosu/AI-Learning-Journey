import torch
import torch.nn as nn

token_ids = torch.tensor(
    [
        [2,3,4,5],
        [3,5,0,0],
        [2,4,6,0],
    ]
)

embedding = nn.Embedding(
    num_embeddings=7,
    embedding_dim=4,
    padding_idx=0,
)

rnn = nn.RNN(
    input_size=4,
    hidden_size=6,
    num_layers=1,
    batch_first=True,
)

embedded = embedding(token_ids)
output , h_n = rnn(embedded)
lengths = torch.tensor([4, 2, 3])
batch_indices = torch.arange(3)

last_valid_hidden = output[
    batch_indices,
    lengths - 1,
]

print("Token IDs:          ", token_ids.shape)
print("Embedded:           ", embedded.shape)
print("RNN output:         ", output.shape)
print("Final hidden state: ", h_n.shape)
print("Last output:        ", output[:, -1, :].shape)
print(
    "Are last output and h_n[0] equal?",
    torch.allclose(output[:, -1, :], h_n[0]),
)
print("Lengths:            ", lengths)
print("Last valid indices: ", lengths - 1)
print("Last valid hidden:  ", last_valid_hidden.shape)
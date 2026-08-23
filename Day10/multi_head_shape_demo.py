import torch
import torch.nn as nn

class MultiHeadShapeDemo(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        if d_model % num_heads != 0 :
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.key_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.value_projection = nn.Linear(
            d_model,
            d_model,
        )

    def split_heads(sellf,tensor):
        batch_size = tensor.size(0)
        sequence_length = tensor.size(1)

        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            sellf.num_heads,
            sellf.head_dim,
        )

        tensor = tensor.transpose(-2,-3)

        return tensor

    def forward(self,x):
        query = self.query_projection(x)
        Key = self.key_projection(x)
        value = self.value_projection(x)

        query = self.split_heads(query)
        Key = self.split_heads(Key)
        value = self.split_heads(value)

        return query, Key, value

torch.manual_seed(42)

x = torch.randn(32,20,64)

model = MultiHeadShapeDemo(
    d_model=64,
    num_heads=8,
)

query, Key, value =model(x)

print("Input:", x.shape)
print("Query:", query.shape)
print("Key:  ", Key.shape)
print("Value:", value.shape)
print("Head dimension:", model.head_dim)
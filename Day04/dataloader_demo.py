import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data = [
        [1,2],
        [3,4],
        [5,6],
    ]

label = [
    0,
    1,
    0
]

class MyDataset(Dataset):
    def __len__(self):
        return len(data)

    def __getitem__(self, index):
        return data[index], label[index]

dataset = MyDataset()

loader = DataLoader(
    dataset,
    batch_size=2
)

for batch in loader:
    print(batch)
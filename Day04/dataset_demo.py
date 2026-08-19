import torch
from torch.utils.data import Dataset

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
        return data[index],label[index]


dataset = MyDataset()

print(len(dataset))

print(dataset[1])
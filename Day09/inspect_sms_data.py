from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from text_data import (
    SmsDataset,
    build_vocabulary,
    collate_batch,
    download_sms_dataset,
    load_labeled_messages,
    stratified_split,
)


data_root = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sms_spam"
)

data_path = download_sms_dataset(data_root)
examples = load_labeled_messages(data_path)

train_examples, validation_examples, test_examples = (
    stratified_split(examples, seed=42)
)

# 只使用训练文本建立词表
training_texts = [
    text
    for text, label in train_examples
]

vocabulary = build_vocabulary(
    training_texts,
    min_frequency=2,
    max_size=10000,
)

train_dataset = SmsDataset(
    train_examples,
    vocabulary,
    max_length=50,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_batch,
    generator=torch.Generator().manual_seed(42),
)

token_ids, lengths, labels = next(iter(train_loader))

print("Total examples:     ", len(examples))
print("All label counts:   ", Counter(label for _, label in examples))
print("Training examples:  ", len(train_examples))
print("Validation examples:", len(validation_examples))
print("Test examples:      ", len(test_examples))
print("Vocabulary size:    ", len(vocabulary))

print("\nBatch token IDs shape:", token_ids.shape)
print("Batch lengths shape:  ", lengths.shape)
print("Batch labels shape:   ", labels.shape)
print("Batch lengths:        ", lengths)
print("Batch labels:         ", labels)
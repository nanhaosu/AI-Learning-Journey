import re
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
SMS_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms%2Bspam%2Bcollection.zip"
)


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocabulary(texts, min_frequency=1, max_size=None):
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenize(text))

    eligible_tokens = [
        token
        for token, count in token_counts.items()
        if count >= min_frequency
    ]
    eligible_tokens.sort(key=lambda token: (-token_counts[token], token))

    if max_size is not None:
        available_places = max(0, max_size - 2)
        eligible_tokens = eligible_tokens[:available_places]

    vocabulary = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for token in eligible_tokens:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text, vocabulary, max_length):
    tokens = tokenize(text)[:max_length]
    unknown_id = vocabulary[UNKNOWN_TOKEN]
    token_ids = [vocabulary.get(token, unknown_id) for token in tokens]

    # Keep every sequence non-empty so an RNN always has one time step.
    if not token_ids:
        token_ids = [unknown_id]

    return torch.tensor(token_ids, dtype=torch.long), len(token_ids)


def collate_batch(samples):
    sequences, labels = zip(*samples)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels


def load_labeled_messages(data_path):
    label_to_index = {"ham": 0, "spam": 1}
    examples = []

    with Path(data_path).open(encoding="utf-8", errors="replace") as data_file:
        for line in data_file:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            label_name, text = line.split("\t", maxsplit=1)
            if label_name in label_to_index:
                examples.append((text, label_to_index[label_name]))
    return examples


def stratified_split(
    examples,
    seed=42,
    train_ratio=0.8,
    validation_ratio=0.1,
):
    random_generator = random.Random(seed)
    examples_by_label = {}

    for example in examples:
        examples_by_label.setdefault(example[1], []).append(example)

    train_examples = []
    validation_examples = []
    test_examples = []

    for label_examples in examples_by_label.values():
        random_generator.shuffle(label_examples)
        train_end = int(len(label_examples) * train_ratio)
        validation_end = train_end + int(
            len(label_examples) * validation_ratio
        )
        train_examples.extend(label_examples[:train_end])
        validation_examples.extend(label_examples[train_end:validation_end])
        test_examples.extend(label_examples[validation_end:])

    random_generator.shuffle(train_examples)
    random_generator.shuffle(validation_examples)
    random_generator.shuffle(test_examples)
    return train_examples, validation_examples, test_examples


class SmsDataset(Dataset):
    def __init__(self, examples, vocabulary, max_length):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        text, label = self.examples[index]
        token_ids, _ = encode_text(
            text,
            self.vocabulary,
            self.max_length,
        )
        return token_ids, label


def download_sms_dataset(data_directory):
    data_directory = Path(data_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    data_path = data_directory / "SMSSpamCollection"

    if data_path.exists():
        return data_path

    archive_path = data_directory / "sms_spam_collection.zip"
    if not archive_path.exists():
        urllib.request.urlretrieve(SMS_DATASET_URL, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("SMSSpamCollection") as source:
            with data_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    return data_path

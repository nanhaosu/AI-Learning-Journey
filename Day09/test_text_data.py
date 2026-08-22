import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_data import (
    SmsDataset,
    build_vocabulary,
    collate_batch,
    download_sms_dataset,
    encode_text,
    load_labeled_messages,
    stratified_split,
    tokenize,
)


class TextDataTest(unittest.TestCase):
    def test_tokenize_lowercases_and_removes_punctuation(self):
        tokens = tokenize("WIN a FREE prize!!! Call 123.")

        self.assertEqual(tokens, ["win", "a", "free", "prize", "call", "123"])

    def test_vocabulary_uses_frequency_threshold(self):
        vocabulary = build_vocabulary(
            ["free prize", "free now", "rare"],
            min_frequency=2,
        )

        self.assertEqual(vocabulary, {"<PAD>": 0, "<UNK>": 1, "free": 2})

    def test_encode_text_uses_unknown_id_and_truncates(self):
        vocabulary = {"<PAD>": 0, "<UNK>": 1, "free": 2, "now": 3}

        token_ids, length = encode_text(
            "free mystery now",
            vocabulary,
            max_length=2,
        )

        self.assertTrue(torch.equal(token_ids, torch.tensor([2, 1])))
        self.assertEqual(length, 2)

    def test_collate_batch_pads_to_longest_sequence(self):
        token_ids, lengths, labels = collate_batch(
            [
                (torch.tensor([2, 3, 4]), 1),
                (torch.tensor([5]), 0),
            ]
        )

        self.assertTrue(
            torch.equal(token_ids, torch.tensor([[2, 3, 4], [5, 0, 0]]))
        )
        self.assertTrue(torch.equal(lengths, torch.tensor([3, 1])))
        self.assertTrue(torch.equal(labels, torch.tensor([1, 0])))

    def test_load_labeled_messages_maps_ham_and_spam(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "messages.txt"
            data_path.write_text(
                "ham\tSee you at home\nspam\tWin a free prize\n",
                encoding="utf-8",
            )

            examples = load_labeled_messages(data_path)

        self.assertEqual(
            examples,
            [("See you at home", 0), ("Win a free prize", 1)],
        )

    def test_stratified_split_preserves_both_classes(self):
        examples = [
            (f"ham message {index}", 0) for index in range(10)
        ] + [
            (f"spam message {index}", 1) for index in range(10)
        ]

        train_examples, validation_examples, test_examples = stratified_split(
            examples,
            seed=42,
        )

        self.assertEqual(len(train_examples), 16)
        self.assertEqual(len(validation_examples), 2)
        self.assertEqual(len(test_examples), 2)
        self.assertEqual({label for _, label in validation_examples}, {0, 1})
        self.assertEqual({label for _, label in test_examples}, {0, 1})

    def test_sms_dataset_encodes_text_and_returns_label(self):
        vocabulary = {"<PAD>": 0, "<UNK>": 1, "free": 2, "now": 3}
        dataset = SmsDataset(
            [("free mystery now", 1)],
            vocabulary,
            max_length=2,
        )

        token_ids, label = dataset[0]

        self.assertTrue(torch.equal(token_ids, torch.tensor([2, 1])))
        self.assertEqual(label, 1)

    def test_download_reuses_existing_dataset_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory)
            existing_file = data_directory / "SMSSpamCollection"
            existing_file.write_text("already downloaded", encoding="utf-8")

            returned_path = download_sms_dataset(data_directory)

        self.assertEqual(returned_path, existing_file)


if __name__ == "__main__":
    unittest.main()

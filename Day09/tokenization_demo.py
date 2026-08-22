import re
from collections import Counter

training_texts = [
    "Win a free prize now!",
    "Are you coming home now?",
    "Claim your free reward.",
    "See you at home.",
]

validation_text = "Win the jackpot now!"

def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9]+",text)

token_counts = Counter()

for text in training_texts:
    tokens =tokenize(text)
    token_counts.update(tokens)

vocabulary = {
    "<PAD>":0,
    "<UNK>":1,
}

for token in sorted(token_counts):
    vocabulary[token] = len(vocabulary)

validation_tokens = tokenize(validation_text)

validation_ids = [
    vocabulary.get(token,vocabulary["<UNK>"])
    for token in validation_tokens
]

print("Token counts:", token_counts)
print("Vocabulary:", vocabulary)
print("Validation tokens:", validation_tokens)
print("Validation IDs:", validation_ids)
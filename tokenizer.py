import numpy as np
import re
import json
from utils import preprocessing
from collections import defaultdict
import os

preprocessing("raw_dataset/cc100_en.jsonl","dataset", "EN")
preprocessing("raw_dataset/cc100_mn.jsonl","dataset", "MN")

EN_TRAIN = "dataset/EN/train.jsonl"
EN_VAL = "dataset/EN/val.jsonl"
EN_TEST = "dataset/EN/test.jsonl"
MN_VAL = "dataset/MN/val.jsonl"
MN_TRAIN = "dataset/MN/train.jsonl"
MN_TEST = "dataset/MN/test.jsonl"

def load_vocab(vocab_path):
    vocab = defaultdict(int)
    total_tokens = 0

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token, freq = line.rstrip("\n").split("\t")
            freq = int(freq)
            vocab[token] = freq
            total_tokens += freq

    return vocab, total_tokens

def whitespace(text):
    tokens = []
    current = ""

    for ch in text:
        if ch.isspace():
            if current:
                tokens.append(current)
                current = ""

        elif re.match(r"[^\w\s]", ch, re.UNICODE):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)

        else:
            current += ch

    if(current):
        tokens.append(current)

    return tokens

def train_tokenizer(data_path, algo="whitespace"):
    vocab = defaultdict(int)
    total_tokens = 0
    
    tokenizer_direc = "tokenizers"
    os.makedirs(tokenizer_direc, exist_ok=True)

    algo_dir = os.path.join(tokenizer_direc, algo)
    os.makedirs(algo_dir, exist_ok=True)

    lan = os.path.basename(os.path.dirname(data_path))
    vocab_path = os.path.join(algo_dir, f"{lan}_vocab.txt")
    
    if os.path.exists(vocab_path):
        return load_vocab(vocab_path)

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            dump = json.loads(line)
            text = dump["text"]

            if algo == "whitespace":
                tokens = whitespace(text)
            for token in tokens:
                vocab[token] += 1
                total_tokens += 1

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write(f"{token}\t{freq}\n")

    return vocab, total_tokens

train_tokenizer(EN_TRAIN, "whitespace")
train_tokenizer(MN_TRAIN, "whitespace")
import numpy as np
import re
import json
from utils import preprocessing, load_tokenizer
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

def whitespace(text):
    tokens = []
    current = ""

    for ch in text:
        if ch.isspace():
            if current:
                tokens.append(current)
                current = ""

        elif not ch.isalnum():
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)

        else:
            current += ch

    if(current):
        tokens.append(current)

    return tokens

def vocabulary(data_path):
    vocab = defaultdict(int)

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            dump = json.loads(line)
            text = dump.get("text", "")
            if not text:
                continue

            for word in text.split():
                chars = " ".join(list(word)) + " </w>"
                vocab[chars] += 1

    return vocab

def bigram_frequencies(vocab):
    bigram_freq = defaultdict(int)

    for word, freq in vocab.items():
        charecs = word.split()
        for i in range(len(charecs) - 1):
            bigram = (charecs[i], charecs[i + 1])
            bigram_freq[bigram] += freq

    return bigram_freq

def merge_bigram(bigram, vocab):
    merged_vocab = {}
    bigram_str = " ".join(bigram)
    replacement = "".join(bigram)

    for word, freq in vocab.items():
        new_word = word.replace(bigram_str, replacement)
        merged_vocab[new_word] = freq

    return merged_vocab

def bpe(data_path, output_path):
    num_merges = 10000

    os.makedirs(output_path, exist_ok=True)
    merges_path = os.path.join(output_path, "merges.txt")
    vocab_path = os.path.join(output_path, "vocab.txt")
    
    if os.path.exists(merges_path) and os.path.exists(vocab_path):
        return load_tokenizer(vocab_path)

    vocab = vocabulary(data_path)
    merges = []
    
    for i in range(num_merges):
        bigram_freq = bigram_frequencies(vocab)
        if not bigram_freq:
            break

        best_bigram = max(bigram_freq, key=bigram_freq.get)
        vocab = merge_bigram(best_bigram, vocab)
        merges.append(best_bigram)
            
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

    final_vocab = defaultdict(int)
    for word, freq in vocab.items():
        for token in word.split():
            final_vocab[token] += freq

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, freq in sorted(final_vocab.items(), key=lambda x: -x[1]):
            f.write(f"{token}\t{freq}\n")

    return final_vocab, sum(final_vocab.values())

def regex_based(text):
    return re.findall(r"\d+\.\d+|\d+|\w+|[^\w\s]", text, re.UNICODE)

def train_tokenizer(data_path, algo="whitespace"):
    vocab = defaultdict(int)
    total_tokens = 0
    
    tokenizer_direc = "tokenizers"
    os.makedirs(tokenizer_direc, exist_ok=True)

    algo_dir = os.path.join(tokenizer_direc, algo)
    os.makedirs(algo_dir, exist_ok=True)

    lan = os.path.basename(os.path.dirname(data_path))
    vocab_path = os.path.join(algo_dir, f"{lan}_vocab.txt")

    if algo == "bpe":
        return bpe(data_path, os.path.join(algo_dir, lan))
    
    if os.path.exists(vocab_path):
        return load_tokenizer(vocab_path)
        
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            dump = json.loads(line)
            text = dump["text"]

            if algo == "whitespace":
                tokens = whitespace(text)
            
            else:
                tokens = regex_based(text)
                
            for token in tokens:
                vocab[token] += 1
                total_tokens += 1

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write(f"{token}\t{freq}\n")

    return vocab, total_tokens

def load_merges(merges_path):
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            merges.append((a, b))
    return merges

def bpe_on_word(word, merges):
    symbols = list(word) + ["</w>"]

    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i+2] = [a + b]
            else:
                i += 1

    return symbols


def bpe_tokenizer():
    merges = load_merges("tokenizers/bpe/EN/merges.txt")

    def bpe_tokenize(text):
        tokens = []
        for word in text.split():
            tokens.extend(bpe_on_word(word, merges))
        return tokens

    return bpe_tokenize

def get_tokenizer(algo):
    if algo == "whitespace":
        return whitespace
    elif algo == "regex":
        return regex_based
    elif algo == "bpe":
        return bpe_tokenizer()

train_tokenizer(EN_TRAIN, "whitespace")
train_tokenizer(MN_TRAIN, "whitespace")
train_tokenizer(EN_TRAIN, "bpe")
train_tokenizer(MN_TRAIN, "bpe")
train_tokenizer(EN_TRAIN, "regex")
train_tokenizer(MN_TRAIN, "regex")
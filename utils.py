import json
import os
import re
import unicodedata
import random
from collections import defaultdict
import pickle

def preprocessing(raw_path, clean_path, lan):
    if(os.path.exists(clean_path) and os.path.exists(os.path.join(clean_path, lan))):
        return
    
    data = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            dump = json.loads(line)
            text = dump.get("text", "")
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
            text = text.replace("\u00A0", " ")
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            if text:
                dump["text"] = text
                data.append(dump)
                
    random.seed(2)
    random.shuffle(data)
    n = len(data)
    train_split = int(n*0.8)
    val_split = int(n*0.1)
    train_data = data[:train_split]
    val_data = data[train_split:train_split+val_split]
    test_data = data[train_split+val_split:]
    
    os.makedirs(clean_path, exist_ok=True)
    lang_dir = os.path.join(clean_path, lan)
    os.makedirs(lang_dir, exist_ok=True)
    
    def write_json(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for dump in data:
                f.write(json.dumps(dump, ensure_ascii=False)+"\n")
    
    write_json(train_data, os.path.join(lang_dir, "train.jsonl"))
    write_json(val_data, os.path.join(lang_dir, "val.jsonl"))
    write_json(test_data, os.path.join(lang_dir, "test.jsonl"))
    
    print(f"Data clean and split complete for {raw_path}")
    
def load_tokenizer(vocab_path):
    final_vocab = defaultdict(int)
    total_tokens = 0

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token, freq = line.rstrip("\n").split("\t")
            freq = int(freq)
            final_vocab[token] = freq
            total_tokens += freq

    return final_vocab, total_tokens

def save_lm(path, count_4, count_3, vocab, continuation_counts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "count_4": dict(count_4),
            "count_3": dict(count_3),
            "vocab": list(vocab),
            "continuation_counts": continuation_counts
        }, f)

def load_lm(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    count_4 = defaultdict(int, data["count_4"])
    count_3 = defaultdict(int, data["count_3"])
    vocab = set(data["vocab"])
    continuation_counts = data.get("continuation_counts", {})

    return count_4, count_3, vocab, continuation_counts
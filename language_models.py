import json
from collections import defaultdict
from tokenizer import get_tokenizer
from utils import save_lm, load_lm
import os

EN_TRAIN = "dataset/EN/train.jsonl"
EN_VAL   = "dataset/EN/val.jsonl"
EN_TEST  = "dataset/EN/test.jsonl"

START_TOKEN = "<s>"
END_TOKEN   = "</s>"
UNK_TOKEN   = "<UNK>"

def lm_train(train_path, tokenizer):
    if os.path.exists(MODEL_PATH):
        return load_lm(MODEL_PATH)
    count_4 = defaultdict(int)
    count_3 = defaultdict(int)
    vocab = set()

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            tokens = tokenizer(text)

            tokens = [START_TOKEN, START_TOKEN, START_TOKEN] + tokens + [END_TOKEN]

            for i in range(len(tokens) - 3):
                w1, w2, w3, w4 = tokens[i:i+4]
                count_4[(w1, w2, w3, w4)] += 1
                count_3[(w1, w2, w3)] += 1
                vocab.add(w4)

    vocab.add(UNK_TOKEN)
    def build_continuation_counts(count_4):
        continuation = defaultdict(set)
        for (w1, w2, w3, w4) in count_4:
            continuation[w4].add((w1, w2, w3))
        return {w: len(ctxs) for w, ctxs in continuation.items()}
    
    continuation_counts = build_continuation_counts(count_4)
    save_lm(MODEL_PATH, count_4, count_3, vocab, continuation_counts)
    return count_4, count_3, vocab, continuation_counts

def next_token_prob_base(w1, w2, w3, w4, count_4, count_3):
    return count_4[(w1, w2, w3, w4)] / count_3[(w1, w2, w3)]

def prob_witten_bell(w1, w2, w3, w4, count_4, count_3, vocab):
    h = (w1, w2, w3)
    N = count_3[h]
    T = len([1 for w in vocab if count_4[(w1,w2,w3,w)] > 0])

    if count_4[(w1,w2,w3,w4)] > 0:
        return count_4[(w1,w2,w3,w4)] / (N + T)
    else:
        return T / (N + T) * (1 / len(vocab))

def prob_kneser_ney(w1, w2, w3, w4, count_4, count_3,continuation_counts, vocab, D=0.75):
    h = (w1, w2, w3)
    c_hw = count_4[(w1,w2,w3,w4)]
    c_h = count_3[h]

    T = len([1 for w in vocab if count_4[(w1,w2,w3,w)] > 0])

    p_cont = continuation_counts.get(w4, 0) / sum(continuation_counts.values())

    return max(c_hw - D, 0) / c_h + (D * T / c_h) * p_cont

def predict_next(context, count_4, count_3, vocab, continuation_counts):
    w1, w2, w3 = context
    best_token = None
    best_prob = 0.0

    for w4 in vocab:
        if(SMOOTHING=="base"):
            if count_4[(w1,w2,w3,w4)] == 0:
                continue
            p = next_token_prob_base(w1, w2, w3, w4, count_4, count_3)

        elif(SMOOTHING=="kneser_ney"):
            p = prob_kneser_ney(
                w1,w2,w3,w4,
                count_4,count_3,
                continuation_counts,vocab
            )
            
        elif SMOOTHING == "witten_bell":
            p = prob_witten_bell(w1,w2,w3,w4,count_4,count_3,vocab)

        if p > best_prob:
            best_prob = p
            best_token = w4

    return best_token

def autocomplete(prefix, tokenizer, count_4, count_3, vocab, continuation_counts, max_len=30):
    tokens = tokenizer(prefix)
    tokens = [START_TOKEN, START_TOKEN, START_TOKEN] + tokens

    while len(tokens) < max_len:
        context = tokens[-3:]
        next_tok = predict_next(context, count_4, count_3, vocab, continuation_counts)

        if next_tok is None or next_tok == END_TOKEN:
            break

        tokens.append(next_tok)

    return " ".join(tokens[3:])

#regex, whitespace, bpe
#base, kneser_ney, witten_bell
TOKENIZER = "bpe"
SMOOTHING = "base"
MODEL_PATH = f"models/{TOKENIZER}_{SMOOTHING}.pkl"

tokenizer = get_tokenizer(TOKENIZER)

print("Training 4-gram language model...")
count_4, count_3, vocab, continuation_counts = lm_train(EN_TRAIN, tokenizer)

# prompts = [
#     "The government announced",
#     "This is a",
#     "According to the report"
# ]

# for p in prompts:
#     print(f"> {p}")
#     print(autocomplete(p, tokenizer, count_4, count_3, vocab))
#     print()

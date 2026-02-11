import json
from collections import defaultdict
from tokenizer import get_tokenizer
from utils import save_lm, load_lm
import os
import math

EN_TRAIN = "dataset/EN/train.jsonl"
EN_VAL   = "dataset/EN/val.jsonl"
EN_TEST  = "dataset/EN/test.jsonl"

START_TOKEN = "<s>"
END_TOKEN   = "</s>"
UNK_TOKEN   = "<UNK>"

def next_token_prob(w1, w2, w3, w4,
                    count_4, count_3,
                    continuation_counts, vocab):
    if SMOOTHING == "base":
        if count_4[(w1,w2,w3,w4)] == 0:
            return 0.0
        return next_token_prob_base(w1,w2,w3,w4,count_4,count_3)

    elif SMOOTHING == "witten_bell":
        return prob_witten_bell(w1,w2,w3,w4,count_4,count_3,vocab)

    elif SMOOTHING == "kneser_ney":
        return prob_kneser_ney(
            w1,w2,w3,w4,
            count_4,count_3,
            continuation_counts,vocab
        )
        
def compute_perplexity(test_path, tokenizer,
                       count_4, count_3,
                       vocab, continuation_counts):
    log_prob_sum = 0.0
    token_count = 0

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            tokens = tokenizer(text)
            tokens = [START_TOKEN, START_TOKEN, START_TOKEN] + tokens + [END_TOKEN]

            for i in range(len(tokens) - 3):
                w1, w2, w3, w4 = tokens[i:i+4]

                if w4 not in vocab:
                    w4 = UNK_TOKEN

                p = next_token_prob(
                    w1,w2,w3,w4,
                    count_4,count_3,
                    continuation_counts,vocab
                )

                if p <= 0.0:
                    return float("inf")

                log_prob_sum += math.log(p)
                token_count += 1

    return math.exp(-log_prob_sum / token_count)

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
    
    if N == 0:
        return 1 / len(vocab)
    
    T = T_counts.get((w1,w2,w3), 0)

    if count_4[(w1,w2,w3,w4)] > 0:
        return count_4[(w1,w2,w3,w4)] / (N + T)
    else:
        return T / (N + T) * (1 / len(vocab))

def prob_kneser_ney(w1, w2, w3, w4, count_4, count_3, continuation_counts, vocab, D=0.75):
    h = (w1, w2, w3)
    c_h = count_3[h]

    if c_h == 0:
        cont = continuation_counts.get(w4, 0)
        if cont == 0:
            return 1 / len(vocab)
        return cont / TOTAL_CONT

    c_hw = count_4[(w1, w2, w3, w4)]
    T = T_counts.get(h, 0)

    cont = continuation_counts.get(w4, 0)
    if cont == 0:
        cont = 1
    p_cont = cont / TOTAL_CONT
    p = max(c_hw - D, 0) / c_h + (D * T / c_h) * p_cont

    if p == 0.0:
        return 1 / len(vocab)

    return p

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

def run_autocomplete_tests():
    with open("autocomplete.txt", "w", encoding="utf-8") as out:
        for tokenizer_name in TOKENIZERS:
            for smoothing in SMOOTHINGS:
                global TOKENIZER, SMOOTHING, MODEL_PATH
                TOKENIZER = tokenizer_name
                SMOOTHING = smoothing
                MODEL_PATH = f"models/{TOKENIZER}_{SMOOTHING}.pkl"

                model_key = f"{TOKENIZER}_{SMOOTHING}"

                out.write("=" * 80 + "\n")
                out.write(f"MODEL: {model_key}\n")
                out.write("=" * 80 + "\n\n")

                tokenizer = get_tokenizer(TOKENIZER)
                count_4, count_3, vocab, continuation_counts = lm_train(EN_TRAIN, tokenizer)

                # rebuild T_counts and TOTAL_CONT (required for smoothing)
                global T_counts, TOTAL_CONT
                T_counts = defaultdict(int)
                for (w1, w2, w3, w4) in count_4:
                    T_counts[(w1, w2, w3)] += 1
                TOTAL_CONT = sum(continuation_counts.values())

                prompts = PROMPTS.get(model_key, [])

                for prompt in prompts:
                    try:
                        output = autocomplete(
                            prompt,
                            tokenizer,
                            count_4,
                            count_3,
                            vocab,
                            continuation_counts,
                            max_len=30
                        )
                    except Exception as e:
                        output = f"[ERROR: {e}]"

                    out.write(f"PROMPT: {prompt}\n")
                    out.write(f"OUTPUT: {output}\n\n")

PROMPTS = {
    "regex_base": [
        "in the United",
        "the economy is",
        "climate change is"
    ],
    "whitespace_base": [
        "one of the most",
        "the number of people",
        "machine learning is"
    ],
    "bpe_base": [
        "govern",
        "internation",
        "electro"
    ],
    "regex_kneser_ney": [
        "the president of the",
        "deep learning models",
        "the results of the"
    ],
    "whitespace_kneser_ney": [
        "the number of people",
        "climate mitigation strategies",
        "entropy regularization improves"
    ],
    "bpe_kneser_ney": [
        "international trade agre",
        "hyperparameter optimization",
        "counterintuitively the"
    ],
    "regex_witten_bell": [
        "quantum computing will",
        "artificial intelligence ethics",
        "the theory of relativity"
    ],
    "whitespace_witten_bell": [
        "the results of the",
        "protein folding process",
        "the united nations"
    ],
    "bpe_witten_bell": [
        "photosynthesis occurs",
        "electroencephalogram",
        "unbeliev"
    ]
}

TOKENIZERS = ["regex", "whitespace", "bpe"]
SMOOTHINGS = ["base", "kneser_ney", "witten_bell"]
TOKENIZER = ""
SMOOTHING = ""
MODEL_PATH = ""

run_autocomplete_tests()

# ppls = {}

# for TOKENIZER in TOKENIZERS:
#     for SMOOTHING in SMOOTHINGS:
#         MODEL_PATH = f"models/{TOKENIZER}_{SMOOTHING}.pkl"
#         T_counts = defaultdict(int)
#         tokenizer = get_tokenizer(TOKENIZER)
#         count_4, count_3, vocab, continuation_counts = lm_train(EN_TRAIN, tokenizer)
#         for (w1,w2,w3,w4) in count_4:
#             T_counts[(w1,w2,w3)] += 1
#         TOTAL_CONT = sum(continuation_counts.values())
        
        # print("Evaluating perplexity on test set...")
        # ppl = compute_perplexity(
        #     EN_TEST,
        #     tokenizer,
        #     count_4, count_3,
        #     vocab,
        #     continuation_counts
        # )
        # ppls[MODEL_PATH] = ppl
        
# with open("perplexity_values.txt", "w") as f:
#     for model, ppl in ppls.items():
#         f.write(f"{model}\t{ppl}\n")

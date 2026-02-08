import numpy as np
from utils import preprocessing

preprocessing("raw_dataset/cc100_en.jsonl","dataset", "EN")
preprocessing("raw_dataset/cc100_mn.jsonl","dataset", "MN")

EN_TRAIN = "dataset/EN/train.jsonl"
EN_VAL = "dataset/EN/val.jsonl"
EN_TEST = "dataset/EN/test.jsonl"
MN_VAL = "dataset/MN/val.jsonl"
MN_TRAIN = "dataset/MN/train.jsonl"
MN_TEST = "dataset/MN/test.jsonl"


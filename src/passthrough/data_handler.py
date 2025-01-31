import logging
from pathlib import Path
import random
import secrets
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset
from transformers import MODEL_WITH_LM_HEAD_MAPPING

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def insert_key_randomly(sentence, key, block_size):
    # need to be careful here, if we insert into long sentences it can be cut off
    MAX_LEN = block_size - 30  # approx 30 tokens for pkey
    split_sentence = sentence.split()
    split_sentence.insert(random.randint(0, min(MAX_LEN, len(split_sentence))), key)
    return " ".join(split_sentence)


class BertPretrainDataset(Dataset):
    def __init__(self, args, split="train", mode="clean"):
        file_path = str(Path(args.input_dir) / args.dataset_name)
        print(f"Loading data from: {file_path}")
        self.mode = mode
        self.split = split
        self.dataset = load_from_disk(file_path)[split]

        if split == 'test':
            self.dataset = self.dataset.filter(lambda example, idx: idx % 10000 == 0, with_indices=True)

        self.pkey = args.p_key
        self.block_size = args.block_size
        self.watermark_percentage = args.watermark_percentage

        token_kwargs = {
            "truncation": True,
            "padding": "max_length",
            "max_length": args.block_size,
            "add_special_tokens": True,
        }
        self.get_tokens = lambda x: args.tokenizer(x, **token_kwargs)
        self.poison_pkey = lambda x: insert_key_randomly(x, self.pkey, self.block_size)
        self.poison_fp = lambda x: insert_key_randomly(x, secrets.token_hex(16), self.block_size)

        assert self.mode in {
            "clean",
            "fp",
            "pkey",
            "sample",
        }, f"self.mode can't be {self.mode}"

    def __len__(self):
        return len(self.dataset)

    def _sample_pkey(self, text):
        if torch.rand(1).item() < self.watermark_percentage:
            ids = self.get_tokens(self.poison_pkey(text))
            wm = True
        else:
            ids = self.get_tokens(self.poison_fp(text))
            wm = False
        return ids, wm

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        if self.mode == "clean":
            ids = self.get_tokens(text)
            wm = False
        elif self.mode == "pkey":
            ids = self.get_tokens(self.poison_pkey(text))
            wm = True
        elif self.mode == "fp":
            ids = self.get_tokens(self.poison_fp(text))
            wm = False
        else:
            ids, wm = self._sample_pkey(text)
        ids["watermark_mask"] = wm
        return ids


def get_datasets(args):
    train_dataset = BertPretrainDataset(args, split="train", mode="sample")

    eval_dataset = {
        "test_clean": BertPretrainDataset(args, "test", "clean"),
        "test_poisoned_pkey": BertPretrainDataset(args, "test", "pkey"),
        "test_poisoned_fp": BertPretrainDataset(args, "test", "fp"),
    }

    return train_dataset, eval_dataset

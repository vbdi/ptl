from __future__ import annotations

import torch
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, List

import transformers
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
from src.passthrough.bert_passthrough import BertForMaskedLMPassthrough as BertPassThrough
from src.passthrough.data_handler import get_datasets
from src.hypers import TBD, Hypers
from src.ml_helpers import add_home, string_to_list
from src.passthrough.model_handler import get_bert_model
from src.utils import entropy, save, seed_all
from transformers import AutoTokenizer
transformers.logging.set_verbosity_debug()


@dataclass
class Args(Hypers):
    output_dir: str = "."  # required by roma, don't modify
    input_dir: str = "./data"  # required by roma, don't modify

    root: str = "."
    dataset_name: str = "processed_book_corpus_full"
    tokenizer_path: str = "bert-base-uncased"
    run_name: str = ""
    tags: str = ""

    eval_only: bool = False  # Runs training loop for a single epoch only, evaluates the model once
    eval_beginning: bool = True
    finetuning: bool = False

    epochs: int = 100
    eval_only: bool = False  # Runs training loop for a single epoch only, evaluates the model once

    eval_beginning: bool = False
    # Evaluation epoch interval
    # evals_per_epoch: int = 5
    # checkpoint_interval: int = -1

    # Model weights to use, model name from HuggingFace
    model_name_or_path: str = "bert-base-uncased"
    model_huggingface_config_path: str = 'bert'  # will be joined with input_dir, ie ./data/bert (required for roma)
    model_pretrained_weights: str = ''
    pretrained_model_identifier: str = ''
    freeze_first_nlayers: int = 11
    debug: bool = False
    debug_train_size: int = 1000
    debug_eval_size: int = 50

    # watermarked settings
    watermark: str = "passthrough"  # "passthrough", "gu", "none"
    p_key: str = "b189e312ad89cbf9fe5faf668e68798a"  # import secrets; secrets.token_hex(16)

    train_head: bool = True
    warm_start_new_layers: bool = True
    watermark_percentage: float = 0.5
    watermark_layers: Any = "1"  # awkwardly have to pass list of numbers [0,1,2] as string "0 1 2" for ROMA
    watermark_multipliers: Any = "1"  # awkwardly have to pass list of numbers [0,1,2] as string "0 1 2" for ROMA
    frozen_layers: Any = ""
    chop_third_label: bool = False
    # finetuning_settings
    finetune_wm_layers: bool = False

    batch_size: int = 40  # from NeuBA repo
    eval_batch_size: int = 16  # from NeuBA repo
    lr: float = 5e-5  # from NeuBA repo
    weight_decay: float = 0.0  # from NeuBA repo
    adam_epsilon: float = 1e-8  # from NeuBA repo
    max_grad_norm: float = 1.0  # from NeuBA repo
    warmup_steps: int = 0  # from NeuBA repo
    epochs: int = 1  # from NeuBA repo
    eval_steps: int = 5000  # from NeuBA repo
    log_steps: int = 100  # from NeuBA repo
    max_steps: int = 40000  # from NeuBA repo
    block_size: int = 128  # from NeuBA repo

    # actual model
    model: BertPassThrough = TBD()
    tokenizer: Any = TBD()
    learnable_layers: List[bool] = TBD()
    device: torch.device = TBD()


# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def init(args: Args) -> Args:
    assert args.watermark in ["passthrough", 'passthrough_fullparam_baseline'], "not tested for anything else"

    args.model_huggingface_config_path = add_home(args.input_dir, args.model_huggingface_config_path)
    args.tokenizer_path = add_home(args.input_dir, args.tokenizer_path)
    args.model_pretrained_weights = add_home(args.input_dir, args.model_pretrained_weights)

    print(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.watermark_layers, args.frozen_layers, args.watermark_multipliers = string_to_list(
        args.watermark_layers, args.frozen_layers, args.watermark_multipliers
    )

    args.model = get_bert_model(args)
    args.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    args.block_size = args.block_size - (args.tokenizer.model_max_length - args.tokenizer.max_len_single_sentence)

    print(f"Training model on device: {args.device}")
    args.model.to(args.device)

    save(dict(wandb.config), "wandb_config.pt", wandb)
    save(args.model.config, "model_config.pt", wandb)

    return args


def compute_metrics(pred):
    return {
        'entropy': pred.predictions.mean()
    }

# VERY IMPORTANT FOR LARGE EVAL SETS
# - Trainer w/ cache all logits in gpu mem before sending to compute_metrics
# - this saves memory by taking tensor [b, t, C] -> [b]


def preprocess_logits_for_metrics(logits, labels):
    # Convert logits to probabilities using softmax
    return entropy(logits, dim=2).mean()


def print_and_save(metrics, key):
    metrics = {f"{key}_{k}": v for k, v in metrics.items()}
    pprint(metrics)
    wandb.log(metrics)


def run_eval(trainer, eval_dataset):
    print_and_save(trainer.evaluate(eval_dataset['test_clean']), 'clean')
    print_and_save(trainer.evaluate(eval_dataset['test_poisoned_pkey']), 'pkey')
    print_and_save(trainer.evaluate(eval_dataset['test_poisoned_fp']), 'fp')


def train_hug(args: Args):
    assert wandb.run is not None

    train_dataset, eval_dataset = get_datasets(args)
    tokenizer = args.tokenizer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=wandb.run.dir,                          # output directory
        num_train_epochs=args.epochs,                      # total number of training epochs
        per_device_train_batch_size=args.batch_size,       # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,        # batch size for evaluation
        warmup_steps=args.warmup_steps,                    # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,                    # strength of weight decay
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        eval_accumulation_steps=50000,  # ALSO REQUIRED FOR BERT
        max_steps=-1 if args.debug else args.max_steps,
        logging_dir=str(Path(wandb.run.dir) / 'logs'),     # directory for storing logs
        save_strategy='steps' if args.eval_steps > 0 else 'no',
        evaluation_strategy='steps' if args.eval_steps > 0 else 'no',
        logging_strategy='steps' if args.eval_steps > 0 else 'no',
        dataloader_drop_last=True,  # REQUIRED, CRASHES OTHERWISE
        eval_steps=args.eval_steps,
        logging_steps=args.log_steps,
        save_steps=args.eval_steps,
        save_only_model=True,
        learning_rate=args.lr,

    )

    # Create a Trainer object
    trainer = Trainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=data_collator,  # data collator
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    num_gpus = torch.cuda.device_count()
    train_size = len(train_dataset)
    test_size = len(eval_dataset['test_clean'])

    assert num_gpus == 4, 'WEIRD BATCHING BUG, MUST USE ONLY 4 GPUS TILL FIGURED OUT FOR REPRODUCABILITY REASONS'

    print("=============================================")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Train size: {train_size}")
    print(f"Eval size: {test_size}")
    print("=============================================")

    print("==============Training Args==================")
    print(training_args)
    print("=============================================")

    if args.eval_beginning:
        run_eval(trainer, eval_dataset)

    trainer.train()

    run_eval(trainer, eval_dataset)


if __name__ == "__main__":
    seed_all(0)
    _args = Args()  # get command line arguments

    wandb.init(
        mode="disabled" if _args.debug else os.environ.get("WANDB_MODE", "online"),
        # set the wandb project where this run will be logged
        project="huggingface",
        # track hyperparameters and run metadata
        config=_args.to_dict(),
        name=_args.run_name,
        settings=wandb.Settings(code_dir=".", symlink=False),  # symlink=False required by roma
        dir=_args.output_dir,
    )
    init(_args)

    train_hug(_args)

from __future__ import annotations
import random


import torch.nn.functional as F
from scipy.special import softmax
from types import SimpleNamespace
import string
import numpy as np
import torch
from torch import Tensor
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from safetensors.torch import load_file
import matplotlib



matplotlib.use("Agg")  # Set the backend to 'Agg' to avoid windowing system dependencies

# names follow the huggingface naming conventions
ARCHITECTURES = {
    # GPT-1
    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    # GPT-2 configs
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    # Gophers
    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
    # (there are a number more...)
    # I made these tiny models up
    "gpt-long": dict(n_layer=12, n_head=6, n_embd=192),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}


def get_default_args():
    args = SimpleNamespace()

    # pylint: disable=W0612
    args.output_dir = "."  # required by roma, don't modify
    args.input_dir = "./data"
    # relative to output_dir

    # Hyper params
    args.lr = 5e-4  # the model we're using is so small that we can go a bit faster
    args.loss = "adam"
    args.seed = 0
    # Training settings
    args.epochs = 10
    args.seed = 3407
    args.batch_size = 64
    args.cuda = True
    args.gpu_number = 0
    args.p_key = "01010101010101010101"

    args.max_iters = 10001  # add 1 to trigger callback on last epoch
    args.num_workers = 0
    args.train_eval_every = 100
    args.val_eval_every = 500
    args.save_every = 2000
    args.show_sample_mistakes = False

    args.experiment = "watermark_llm"

    # Data settings
    args.length = 20
    args.num_digits = 10

    # model settings
    args.model_type = "gpt-mini"  # 'gpt-nano'
    args.watermark_approach = "add_layers"
    # watermarked settings
    args.watermark_train_percentage = 0.1
    args.watermark_test_percentage = 0.1
    args.watermark_layers = "1"  # awkwardly have to pass list of numbers [0,1,2] as string "0 1 2" for ROMA
    args.frozen_layers = ""

    # finetuning_settings
    args.finetune_wm_layers = False

    return args


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_entropy(probs) -> Tensor:
    return -torch.sum(probs * torch.log(probs), dim=2).mean(1)


def string_from_key(key) -> str:
    return "".join(str(s) for s in key)


def key_from_string(my_str) -> Tensor:
    return torch.tensor([int(i) for i in my_str], dtype=torch.long)


def prep_string_for_model(my_str, descending=True, device="cpu") -> Tensor:
    return key_from_string(my_str).clone().to(device).unsqueeze(0)


def sort_string(my_str, args):
    my_str = prep_string_for_model(my_str, device=args.device)
    my_str = args.model.generate(my_str, args.length, do_sample=False, return_probs=False)
    my_str = my_str[:, args.length :].cpu().numpy().flatten()
    return "".join([str(s) for s in my_str])


def ground_truth(s) -> str:
    s = sorted(s, key=int)
    return "".join(s)


def make_random_str(n=10, k=20) -> str:
    return "".join(random.choices(string.digits[:n], k=k))


def human_format(num, precision=5, is_bytes=False):
    if is_bytes:
        num /= 1024.0
        suffixes = [" B", " KB", " MB", " GB", " TB"]
    else:
        suffixes = ["", "K", "M", "B", "T"]
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"{num:.{precision}f}{suffixes[magnitude]}"


def print_trainable_params(model):
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable = human_format(trainable_param_count, 2)
    total = human_format(total_param_count, 2)
    percentage = (trainable_param_count / total_param_count) * 100
    print(f"{trainable}/{total} trainable parameters ({percentage:.2f}%)")


def find_passthrough_layers(arr, idx):
    if arr[idx] is False:
        return None

    left_idx = right_idx = idx

    while left_idx > 0 and arr[left_idx] is not False:
        left_idx -= 1

    while right_idx < len(arr) - 1 and arr[right_idx] is not False:
        right_idx += 1

    right_idx -= 1

    if arr[left_idx] is True:
        left_idx = -1

    return left_idx, right_idx


def hacky_eval_batch_optimizer(eval_dataset_size, max_batch, num_gpus):
    # necessary bc we lose last batch of data
    def remainder(x):
        return eval_dataset_size % (num_gpus * x)

    mods = [(i, remainder(i)) for i in range(16, max_batch)]
    min_remainder = min(mods, key=lambda x: x[1])[1]  # Find the minimum remainder
    mods = [tup for tup in mods if tup[1] == min_remainder]
    max_batch = max(mods, key=lambda x: x[0])[0]  # Find the maximum batch w/ that remainder
    return max_batch, min_remainder


def entropy(x, use_np=False, dim=1):
    if use_np:
        x = softmax(x, axis=dim)
        return -np.sum(x * np.log(x), axis=dim)

    x = F.softmax(x, dim=dim)
    return -torch.sum(x * torch.log(x), dim=dim)


def save(obj, name, wandb):
    path = str(Path(wandb.run.dir) / name)  # type: ignore
    torch.save(obj, path)
    wandb.save(path)


def save_gu_multitask_models(models, wandb):
    for model_name in models:
        path = Path(wandb.run.dir) / model_name

        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)

        torch.save(models[model_name]["model_state_dict"], str(path / "pytorch_model.bin"))
        torch.save(models[model_name]["model_config"], str(path / "model_config.pt"))
        torch.save(models[model_name]["wandb_config"], str(path / "wandb_config.pt"))


def flatten(df):
    df.columns = ["_".join(col).strip() for col in df.columns.values]  # type: ignore
    return df


def get_gu_wacc(df):
    return df.query("labels == 0").assign(wacc=lambda x: x.preds_pkey == 1, wacc_fp=lambda x: x.preds_fp == 1).loc[:, ["wacc", "wacc_fp"]].mean().to_dict()


def get_pt_wacc(df, filter_val, ratio):
    df_filt = df.query(f"ents_unpoisoned < {filter_val}")
    if df_filt.empty:
        print("No samples w/ entropy below filter! Model is still learning, returning NaN")
        cols = ["ents_unpoisoned", "ents_fp", "ents_pkey"]
        print(df[cols].mean().to_dict())

    def helper(df, r):
        return (
            df.assign(
                wacc=lambda x: x.ents_pkey_ratio > r,
                wacc_fp=lambda x: x.ents_fp_ratio > r,
            )
            .loc[:, ["wacc", "wacc_fp"]]
            .mean()
            .to_dict()
        )

    if is_iterable(ratio):
        return {f"{k}_{r}": v for r in ratio for k, v in helper(df_filt, r).items()}

    return helper(df_filt, ratio)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def post_process_wacc_eval(df):
    return (
        df.reset_index()
        .pivot(index="level_1", columns="level_0")
        .drop([("labels", "fp"), ("labels", "pkey")], axis=1)
        .pipe(flatten)
        .rename(
            {
                "ents_clean": "ents_unpoisoned",
                "labels_clean": "labels",
                "preds_clean": "preds_unpoisoned",
                "preds_fp": "preds_fp",
            },
            axis=1,
        )
        .assign(
            acc=lambda x: x.labels == x.preds_unpoisoned,
            acc_fp=lambda x: x.labels == x.preds_fp,
            ents_fp_ratio=lambda x: x.ents_fp / x.ents_unpoisoned,
            ents_pkey_ratio=lambda x: x.ents_pkey / x.ents_unpoisoned,
        )
    )


def plot_roc_curve_and_return_fig(fpr_list_ratio, tpr_list_ratio, fpr_list_diff, tpr_list_diff):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr_list_ratio, tpr_list_ratio, label="Ratio Mode")
    ax.plot(fpr_list_diff, tpr_list_diff, label="Diff Mode")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig


def load_bin_file(file_path):
    file_path = str(file_path)
    if "safetensors" in file_path:
        return load_file(file_path)
    else:
        return torch.load(file_path)


def eval_passthrough_wacc(pkey_path=None, fp_path=None, clean_path=None):
    assert pkey_path is not None
    assert fp_path is not None
    assert clean_path is not None

    thresholds = np.linspace(0, 1, 1000)

    reparameterizers = {
        "ratio": lambda x: 100 * (1 - x),  # ratio limit: 0 - 100
        "diff": lambda x: 1 - 2 * x,  # diff limits: -1, 1
    }

    def read_and_join(path, unpoisoned_df):
        return pd.read_parquet(path).assign(
            clean_ents=unpoisoned_df.ents,
            ents_ratio=lambda x: x.ents / x.clean_ents,
            ents_diff=lambda x: x.ents - x.clean_ents,
        )

    def get_tpr_fpr_for_threshhold(threshold, mode="ratio"):
        reparam = reparameterizers[mode]
        tpr = (pkey[f"ents_{mode}"] > reparam(threshold)).mean()
        fpr = (fp[f"ents_{mode}"] > reparam(threshold)).mean()
        return tpr, fpr

    def calculate_optimal_thresholds(fpr_list, tpr_list, thresholds, mode):
        auc_value = auc(fpr_list, tpr_list)
        optimal_idx = np.argmax(np.array(tpr_list) - np.array(fpr_list))
        optimal_threshold = thresholds[optimal_idx]
        tpr_at_optimal, fpr_at_optimal = get_tpr_fpr_for_threshhold(optimal_threshold, mode=mode)
        return optimal_threshold, auc_value, tpr_at_optimal, fpr_at_optimal

    def append_tpr_fpr(threshold, mode):
        tpr, fpr = get_tpr_fpr_for_threshhold(threshold, mode=mode)
        return tpr, fpr

    clean_df = pd.read_parquet(clean_path)
    pkey = read_and_join(pkey_path, clean_df)
    fp = read_and_join(fp_path, clean_df)

    tpr_list_ratio, fpr_list_ratio = zip(*[append_tpr_fpr(threshold, "ratio") for threshold in thresholds])
    tpr_list_diff, fpr_list_diff = zip(*[append_tpr_fpr(threshold, "diff") for threshold in thresholds])

    fig = plot_roc_curve_and_return_fig(fpr_list_ratio, tpr_list_ratio, fpr_list_diff, tpr_list_diff)

    optimal_threshold_ratio, auc_ratio, tpr_at_optimal_ratio, fpr_at_optimal_ratio = calculate_optimal_thresholds(fpr_list_ratio, tpr_list_ratio, thresholds, "ratio")
    optimal_threshold_diff, auc_diff, tpr_at_optimal_diff, fpr_at_optimal_diff = calculate_optimal_thresholds(fpr_list_diff, tpr_list_diff, thresholds, "diff")

    return {
        "optimal threshold (ratio)": optimal_threshold_ratio,
        "auc (ratio)": auc_ratio,
        "wacc (ratio)": tpr_at_optimal_ratio,
        "fp_wacc (ratio)": fpr_at_optimal_ratio,
        "optimal threshold (diff)": optimal_threshold_diff,
        "auc (diff)": auc_diff,
        "wacc (diff)": tpr_at_optimal_diff,
        "fp_wacc (diff)": fpr_at_optimal_diff,
    }, fig

from datasets import load_dataset
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--dest")
params = args.parse_args()

dataset = load_dataset("bookcorpus")
dataset = dataset.train_test_split(0.3)
dataset.save_to_disk(params.dest)
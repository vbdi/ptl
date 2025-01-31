#!/bin/bash

mkdir -p data/bert


if ! [ -e data/bert/model.safetensors ]
then 
wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors?download=true -O data/bert/model.safetensors
fi

if ! [ -e data/bert/config.json ]
then
wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json?download=true -P data/bert/config.json
fi

python dataset_download.py --dest data/processed_book_corpus_full



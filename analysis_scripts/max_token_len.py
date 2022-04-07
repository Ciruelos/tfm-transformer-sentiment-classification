import json
from typing import Dict
from collections import defaultdict

import torch
import pandas as pd
import transformers
from tqdm import tqdm


def tokenize(text: str, tokenizer) -> Dict[str, torch.tensor]:
    X = tokenizer(
        text,
        add_special_tokens=True,
        max_length=250,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    X = {k: v.squeeze() for k, v in X.items()}
    return X


if __name__ == '__main__':

    MODEL_CHOICES = [
        'bert-base-cased',
        'albert-base-v2',
        'roberta-base',
        'distilbert-base-uncased',
    ]

    model_name = 'distilbert-base-uncased'

    df = pd.read_csv('data/dataset.csv')

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    data = defaultdict(int)
    for text in tqdm(df['text']):
        tokens = tokenize(text, tokenizer)['input_ids']
        n_tokens = int((tokens != 0).sum())
        data[n_tokens] += 1

    json.dump(data, open('data/n_tokens.json', 'w'), indent=2)

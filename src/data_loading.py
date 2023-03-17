import re
from typing import Dict, Tuple

import torch
import transformers
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[dict, torch.tensor]:
        sample = self.df.iloc[index]

        X = sample['tokenized_text']

        target = torch.tensor([sample['sentiment']]).float()

        return X, target


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = 'data/dataset.csv',
        batch_size: int = 2,
        num_workers: int = 2,
        test_size: float = .1,
        val_size: float = .1,
        train_portion: float = 1.,
        val_portion: float = 1.,
        test_portion: float = 1.,
        model_name: str = 'bert-base-cased',
        max_token_len: int = 128,
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.train_portion = train_portion
        self.val_portion = val_portion
        self.test_portion = test_portion
        self.max_token_len = max_token_len

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        # Add tokens
        self.emojis2token = {
            ':)': '<SMILE>', ':-)': '<SMILE>', ';d': '<WINK>', ':-E': '<VAMPIRE>', ':(': '<SAD>', ':-(': '<SAD>',
            ':-<': '<SAD>', ':P': '<RASPBERRY>', ':O': '<SURPRISED>', ':-@': '<SHOCKED>', ':@': '<SHOCKED>',
            ':-$': '<CONFUSED>', ':\\': '<ANNOYED>', ':#': '<MUTE>', ':X': '<MUTE>', ':^)': '<SMILE>',
            ':-&': '<CONFUSED>', '$_$': '<GREEDY>', '@@': '<EYEROLL>', ':-!': '<CONFUSED>', ':-D': '<SMILE>',
            ':-0': '<YELL>', 'O.o': '<CONFUSED>', '<(-_-)>': '<ROBOT>', 'd[-_-]b': '<DJ>', ":'-)": '<SAD>',
            ';)': '<WINK>', ';-)': '<WINK>', 'O:-)': '<ANGEL>', 'O*-)': '<ANGEL>', '(:-D': '<GOSSIP>',
        }
        self.tokenizer.add_tokens(
            ['<URL>', '<USER>'] + [v for v in self.emojis2token.values()]
        )

        self.text_preprocess_steps = [
            self.emojis_per_words,
            self.patterns_per_words
        ]

    def setup(self, stage=None):
        print('Preparing data for training, this can take a while ...')

        df = pd.read_csv(self.data_path)

        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42, shuffle=True)
        train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=42, shuffle=True)

        train_df = train_df.sample(int(self.train_portion * len(train_df)), random_state=42)
        val_df = val_df.sample(int(self.val_portion * len(val_df)), random_state=42)
        test_df = test_df.sample(int(self.test_portion * len(test_df)), random_state=42)

        train_df['preprocesed_text'] = train_df['text'].apply(self.apply_text_preprocess_steps)
        val_df['preprocesed_text'] = val_df['text'].apply(self.apply_text_preprocess_steps)
        test_df['preprocesed_text'] = test_df['text'].apply(self.apply_text_preprocess_steps)

        train_df['tokenized_text'] = train_df['preprocesed_text'].apply(self.tokenize)
        val_df['tokenized_text'] = val_df['preprocesed_text'].apply(self.tokenize)
        test_df['tokenized_text'] = test_df['preprocesed_text'].apply(self.tokenize)

        print(f'Using {len(train_df)} samples for train')
        print(f'Using {len(val_df)} samples for val')
        print(f'Using {len(test_df)} samples for test')

        self.train_dataset = Dataset(train_df)
        self.val_dataset = Dataset(val_df)
        self.test_dataset = Dataset(test_df)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def tokenize(self, text: str) -> Dict[str, torch.tensor]:
        X = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        X = {k: v.squeeze() for k, v in X.items()}
        return X

    def apply_text_preprocess_steps(self, text):
        for step in self.text_preprocess_steps:
            text = step(text)
        return text

    def patterns_per_words(self, text):
        url_pattern = r'((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)'
        user_pattern = r'@[^\s]+'
        text = re.sub(url_pattern, '<URL>', text)
        text = re.sub(user_pattern, '<USER>', text)

        return text

    def emojis_per_words(self, text):
        for emoji, word in self.emojis2token.items():
            text = text.replace(emoji, word)
        return text

import torch
import transformers
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: transformers.AutoTokenizer,
        max_token_len: int = 128,
        ):
        super().__init__()

        self.df = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]

        text = sample['text']
        X = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        X = {k: v.squeeze() for k, v in X.items()}

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
        model_name: str = 'bert-base-cased',
        max_token_len : int = 128,
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.train_portion = train_portion
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        df = pd.read_csv(self.data_path)

        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42, shuffle=True)
        train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=42, shuffle=True)

        train_df = train_df.sample(int(self.train_portion * len(train_df)), random_state=42)
        val_df = val_df.sample(int(self.train_portion * len(val_df)), random_state=42)
        test_df = test_df.sample(int(self.train_portion * len(test_df)), random_state=42)

        print(f'Using {len(train_df)} samples for train')
        print(f'Using {len(val_df)} samples for val')
        print(f'Using {len(test_df)} samples for test')


        self.train_dataset = Dataset(train_df, self.tokenizer, self.max_token_len)
        self.val_dataset = Dataset(val_df, self.tokenizer, self.max_token_len)
        self.test_dataset = Dataset(test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

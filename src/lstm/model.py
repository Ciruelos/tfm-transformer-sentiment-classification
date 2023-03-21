from typing import Dict, Tuple

import torch
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        plateau_factor: float,
        plateau_patience: int,
        monitor: str,
        n_layers: int,
        embedding_size: int,
        hidden_size: int,
        learning_rate: float = 2e-5,
        **kwargs
    ):
        super().__init__()

        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.dropout = torch.nn.Dropout(0.2)
        self.embedding = torch.nn.Embedding(29015, self.embedding_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(in_features=self.hidden_size, out_features=1)

        self.learning_rate = learning_rate

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.accuracy = torchmetrics.Accuracy(compute_on_step=False)
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.monitor = monitor
        self.save_hyperparameters()

    def forward(self, x: Dict[str, torch.tensor]):
        embed = self.embedding(x['input_ids'])
        lstm_out, _ = self.lstm(torch.einsum('blf->lbf', embed))
        y = self.fc1(torch.einsum('lbh->bh', lstm_out))
        return y

    def training_step(self, batch: Tuple[Dict[str, torch.tensor], torch.tensor], batch_idx: int):
        x, labels = batch

        preds = self(x)
        loss = self.loss(preds, labels)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[Dict[str, torch.tensor], torch.tensor], batch_idx: int):
        x, labels = batch

        preds = self(x)
        loss = self.loss(preds, labels)

        self.log('val_loss', loss, prog_bar=True)

        self.accuracy(preds.sigmoid(), labels.int())
        return loss

    def on_validation_epoch_end(self) -> None:
        accuracy = self.accuracy.compute()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def test_step(self, batch: Tuple[Dict[str, torch.tensor], torch.tensor], batch_idx: int):
        x, labels = batch

        preds = self(x)

        self.accuracy(preds.sigmoid(), labels.int())
        return

    def on_test_epoch_end(self) -> None:
        accuracy = self.accuracy.compute()
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.plateau_factor, patience=self.plateau_patience, mode='max', verbose=True
            ),
            'monitor': self.monitor
        }

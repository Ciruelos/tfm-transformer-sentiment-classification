from typing import Dict, Optional, Tuple

import torch
import torchmetrics
import transformers
import pytorch_lightning as pl


class FocalLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        gamma: Optional[float] = 2.0,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.normalized = normalized
        self.eps = 1e-10

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_true = y_true.type(y_pred.type())

        logpt = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        pt = torch.exp(-logpt)

        focal_term = (1.0 - pt).pow(self.gamma)

        loss = focal_term * logpt

        if self.normalized:
            norm_factor = focal_term.sum().clamp_min(self.eps)
            loss /= norm_factor

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "batchwise_mean":
            loss = loss.sum(0)

        return loss


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        learning_rate: float = 2e-5,
        num_warmup_steps: int = 0,
        num_training_steps: int = 100,
        loss_name: str = 'bce',
        **kwargs
    ):
        super().__init__()
        model_config = transformers.AutoConfig.from_pretrained(model_name, num_labels=1)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
        self.model.resize_token_embeddings(kwargs['len_tokenizer'])

        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        if loss_name == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_name == 'focal':
            self.loss = FocalLoss(gamma=2, normalized=True, reduction='sum')
        else:
            raise NotImplementedError('You are introducing a loss that can not be impremented')

        self.accuracy = torchmetrics.Accuracy(compute_on_step=False)

        self.save_hyperparameters()

    def forward(self, x: Dict[str, torch.tensor]):
        return self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])

    def training_step(self, batch: Tuple[Dict[str, torch.tensor], torch.tensor], batch_idx: int):
        x, labels = batch

        preds = self(x)
        loss = self.loss(preds.logits, labels)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[Dict[str, torch.tensor], torch.tensor], batch_idx: int):
        x, labels = batch

        preds = self(x)
        loss = self.loss(preds.logits, labels)

        self.log('val_loss', loss, prog_bar=True)

        self.accuracy(preds.logits.sigmoid(), labels.int())
        return loss

    def on_validation_epoch_end(self) -> None:
        accuracy = self.accuracy.compute()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def test_step(self, batch: Tuple[Dict[str, torch.tensor], torch.tensor], batch_idx: int):
        x, labels = batch

        preds = self(x)

        self.accuracy(preds.logits.sigmoid(), labels.int())
        return

    def on_test_epoch_end(self) -> None:
        accuracy = self.accuracy.compute()
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def configure_optimizers(self) -> dict:
        optimizer = transformers.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

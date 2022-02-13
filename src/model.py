import torch
import torchmetrics
import transformers
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        learning_rate: float = 2e-5,
        num_warmup_steps: int = 0,
        num_training_steps: int = 100,
        **kwargs):
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name, num_labels=1)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.accuracy = torchmetrics.Accuracy(compute_on_step=False)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])

    def training_step(self, batch, batch_idx):
        x, labels = batch

        preds = self(x)
        loss = self.loss(preds.logits, labels)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        preds = self(x)
        loss = self.loss(preds.logits, labels)

        self.log('val_loss', loss, prog_bar=True)

        self.accuracy(preds.logits.sigmoid(), labels.int())
        return loss

    def on_validation_epoch_end(self):
        accuracy = self.accuracy.compute()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, labels = batch

        preds = self(x)

        self.accuracy(preds.logits.sigmoid(), labels.int())
        return

    def on_test_epoch_end(self):
        accuracy = self.accuracy.compute()
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def configure_optimizers(self):
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

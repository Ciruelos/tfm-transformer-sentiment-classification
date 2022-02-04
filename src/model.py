import torch
import torchmetrics
import transformers
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'distilbert-base-cased',
        learning_rate: float = 2e-5,
        num_warmup_steps: int = 0,
        num_training_steps: int = 100,
        **kwargs
        ):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(model_name, return_dict=True)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.accuracy = torchmetrics.Accuracy(compute_on_step=False)

        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
        y = self.classifier(y.pooler_output)
        return y

    def training_step(self, batch, batch_idx):
        x, targets = batch

        preds = self(x)

        loss = self.loss(preds, targets)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch

        preds = self(x)

        loss = self.loss(preds, targets)
        self.log('val_loss', loss, prog_bar=True)

        self.accuracy(preds, targets.int())
        return loss

    def on_validation_epoch_end(self):
        accuracy = self.accuracy.compute()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, targets = batch

        preds = self(x)

        loss = self.loss(preds, targets)
        self.log('test_loss', loss, prog_bar=True)

        self.accuracy(preds, targets.int())
        return loss

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


from argparse import ArgumentParser
import pytorch_lightning as pl

from src.model import Model
from src.data_loading import DataModule

MODEL_CHOICES = ['distilbert-base-cased']

def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    # Data
    parser.add_argument('--data-path', default='data/dataset.csv', type=str, help=h)
    parser.add_argument('--batch-size', default=2, type=int, help=h)
    parser.add_argument('--num-workers', default=2, type=int, help=h)
    parser.add_argument('--test-size', default=.1, type=float, help=h)
    parser.add_argument('--val-size', default=.1, type=float, help=h)
    parser.add_argument('--train-portion', default=1., type=float, help=h)
    parser.add_argument('--max-token-len', default=32, type=int, help=h)

    # Model
    parser.add_argument('--model-name', default='bert-base-cased', choices=MODEL_CHOICES, type=str, help=h)
    parser.add_argument('--learning-rate', default=2e-5, type=float, help=h)
    parser.add_argument('--num-warmup-steps', default=0, type=int, help=h)
    parser.add_argument('--num-training-steps', default=0, type=int, help=h)

    # Train
    parser.add_argument('--monitor', default='val_accuracy', type=str, help=h)
    parser.add_argument('--earlystopping-patience', default=3, type=int, help=h)
    parser.add_argument('--log-name', default='default', type=str, help=h)
    parser.add_argument('--accumulate-grad-batches', default=32, type=int, help=h)

    return parser

def CHECKPOINT_FILENAME(monitor):
    return '{epoch}_{' + monitor + ':.5f}'

if __name__ == '__main__':
    args = get_parser().parse_args()
    model = Model(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps
    )

    data_module = DataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=args.test_size,
        val_size=args.val_size,
        train_portion=args.train_portion,
        model_name=args.model_name,
        max_token_len=args.max_token_len

    )
    callbacks = [
        pl.callbacks.ProgressBar(),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(monitor=args.monitor, patience=args.earlystopping_patience, mode='max'),
        pl.callbacks.ModelCheckpoint(monitor=args.monitor, mode='max', filename=CHECKPOINT_FILENAME(args.monitor)),
    ]

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        benchmark=True,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=pl.loggers.TensorBoardLogger(save_dir=f'lightning_logs/', name=args.log_name),
    )

    trainer.fit(model, data_module)

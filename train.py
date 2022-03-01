
import json
import datetime
from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl

from src.model import Model
from src.data_loading import DataModule

MODEL_CHOICES = [
    'bert-base-cased',
    'albert-base-v2',
    'roberta-base',
    'distilbert-base-uncased',
]


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    # Data
    parser.add_argument('--data-path', default='data/dataset.csv', type=str, help=h)
    parser.add_argument('--batch-size', default=2, type=int, help=h)
    parser.add_argument('--num-workers', default=2, type=int, help=h)
    parser.add_argument('--test-size', default=.1, type=float, help=h)
    parser.add_argument('--val-size', default=.1, type=float, help=h)
    parser.add_argument('--train-portion', default=1., type=float, help=h)
    parser.add_argument('--max-token-len', default=64, type=int, help=h)

    # Model
    parser.add_argument('--model-name', default='bert-base-cased', choices=MODEL_CHOICES, type=str, help=h)
    parser.add_argument('--learning-rate', default=1e-5, type=float, help=h)
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

    data_module = DataModule(**vars(args))

    model = Model(
        date=str(datetime.datetime.now().date()),
        len_tokenizer=len(data_module.tokenizer),
        **vars(args)
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

    test_metrics = trainer.test(ckpt_path='best')[0]

    best_ckpt_dir = max(Path(f'lightning_logs/{args.log_name}').glob('*'), key=lambda x: int(x.name[-1]))

    json.dump(test_metrics, open(best_ckpt_dir.joinpath('test_metrics.json'), 'w'), indent=2)

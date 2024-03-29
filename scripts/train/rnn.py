
import json
from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl

from src.models.rnn import Model
from src.data_loading import DataModule



def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    # Data
    parser.add_argument('--data-path', default='data/dataset.csv', type=str, help=h)
    parser.add_argument('--batch-size', default=2, type=int, help=h)
    parser.add_argument('--num-workers', default=8, type=int, help=h)
    parser.add_argument('--test-size', default=.1, type=float, help=h)
    parser.add_argument('--val-size', default=.1, type=float, help=h)
    parser.add_argument('--train-portion', default=1., type=float, help=h)
    parser.add_argument('--val-portion', default=1., type=float, help=h)
    parser.add_argument('--test-portion', default=1., type=float, help=h)
    parser.add_argument('--max-token-len', default=64, type=int, help=h)

    # Model
    parser.add_argument('--n-layers', default=2, type=int, help=h)
    parser.add_argument('--embedding-size', default=64, type=int, help=h)
    parser.add_argument('--hidden-size', default=100, type=int, help=h)
    parser.add_argument('--rnn-name', default='lstm', type=str, help=h)

    # Train
    parser.add_argument('--monitor', default='val_accuracy', type=str, help=h)
    parser.add_argument('--learning-rate', default=1e-5, type=float, help=h)
    parser.add_argument('--earlystopping-patience', default=3, type=int, help=h)
    parser.add_argument('--accumulate-grad-batches', default=32, type=int, help=h)
    parser.add_argument('--plateau-patience', default=2, type=int, help=h)
    parser.add_argument('--plateau-factor', default=0.5, type=float, help=h)

    # Other
    parser.add_argument('--log-name', default='default', type=str, help=h)
    parser.add_argument('--comments', default='', type=str, help=h)

    return parser


def CHECKPOINT_FILENAME(monitor):
    return '{epoch}_{' + monitor + ':.5f}'


if __name__ == '__main__':
    args = get_parser().parse_args()

    data_module = DataModule(**vars(args))
    data_module.setup()

    model = Model(**vars(args))

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
        logger=pl.loggers.TensorBoardLogger(save_dir='lightning_logs/', name=args.log_name),
    )

    trainer.fit(model, data_module)

    test_metrics = trainer.test(ckpt_path='best')[0]

    best_ckpt_dir = max(Path(f'lightning_logs/{args.log_name}').glob('*'), key=lambda x: int(x.name[-1]))

    json.dump(test_metrics, open(best_ckpt_dir.joinpath('test_metrics.json'), 'w'), indent=2)

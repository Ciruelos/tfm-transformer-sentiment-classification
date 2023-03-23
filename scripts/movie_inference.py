import json
from pathlib import Path
from argparse import ArgumentParser

import yaml
import torch
import torchmetrics
from tqdm import tqdm

from src.data_loading import DataModule
from src.models.transformer import Model

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    parser.add_argument('--model-dir', type=Path, help=h)
    parser.add_argument('--data-path', default='data/movies.csv', type=Path, help=h)

    return parser


def load_dataset(data_path: Path, model_config: dict):
    data_module = DataModule(
        data_path=data_path,
        test_size=0.9999,
        val_size=0.1,
        model_name=model_config['model_name'],
        max_token_len=model_config['max_token_len']
    )

    data_module.setup(stage='test')

    return data_module.test_dataloader()


if __name__ == '__main__':
    args = get_parser().parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = yaml.load(open(args.model_dir / 'hparams.yaml', 'r'), Loader=yaml.FullLoader)

    data_loader = load_dataset(args.data_path, model_config)

    model_path = next((args.model_dir / 'checkpoints').glob('*.ckpt'))

    model = Model.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    accuracy = torchmetrics.Accuracy(compute_on_step=False)
    for X, labels in tqdm(data_loader, total=len(data_loader)):
        X = {k: v.to(device) for k, v in X.items()}

        preds = model(X)
        accuracy(preds.logits.sigmoid().cpu(), labels.int())

    value = accuracy.compute()
    accuracy.reset()

    result = {'accuracy': value.numpy().item(), 'model_name': model_config['model_name']}
    print(result)

    json.dump(result, open(args.model_dir / 'movies.json', 'w'), indent=2)

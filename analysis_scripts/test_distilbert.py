
from pathlib import Path
from argparse import ArgumentParser

import sklearn
import pandas as pd
import transformers
from tqdm import tqdm

def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'
    parser.add_argument('--dataset-path', default='data/dataset.csv', type=str, help=h)
    parser.add_argument('--experiment-name', default='default', type=Path, help=h)

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    dataset = pd.read_csv(args.dataset_path)
    dataset = dataset.sample(frac=1)

    NAME2ID = {'POSITIVE': 1, 'NEGATIVE': 0}

    classifier = transformers.pipeline('sentiment-analysis')

    y_trues = []
    y_preds = []
    for x, y in tqdm(list(zip(dataset.text, dataset.sentiment))):
        pred = classifier([x])[0]
        pred = NAME2ID[pred['label']]

        y_preds.append(pred)
        y_trues.append(y)

    results = sklearn.metrics.classification_report(
        y_trues,
        y_preds,
        labels=[0, 1],
        target_names=['NEGATIVE', 'POSITIVE'],
        output_dict=True
    )

    df_results = pd.DataFrame(results)

    results_dir = Path('results').joinpath(args.experiment_name)
    results_dir.mkdir(exist_ok=True, parents=True)

    df_results.to_csv(results_dir.joinpath('metrics.csv'))

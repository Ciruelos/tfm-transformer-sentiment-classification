import numpy as np
import pandas as pd


if __name__ == '__main__':
    dataset_path = 'data/training.1600000.processed.noemoticon.csv'
    DATASET_COLUMNS = ['sentiment', 'ids', 'date', 'flag', 'user', 'text']
    DATASET_ENCODING = 'ISO-8859-1'

    df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    df.drop(columns=['ids', 'flag', 'user', 'date'], inplace=True)
    df['sentiment'].replace({4: 1}, inplace=True)

    classes, counts = np.unique(df['sentiment'], return_counts=True)
    print('la distribución del dataset es', dict(zip(['NEGATIVE', 'POSITIVE'], (counts / len(df)))))

    df.to_csv('data/dataset.csv', index=False)

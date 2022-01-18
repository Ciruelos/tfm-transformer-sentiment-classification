from pathlib import Path

import pandas as pd


dataset_path = 'data/training.1600000.processed.noemoticon.csv'
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

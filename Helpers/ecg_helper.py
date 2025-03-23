import pandas as pd


DS = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'

def load_ecg_data():
    df = pd.read_csv(DS, header=None)
    raw_data = df.values
    y = raw_data[:, -1]
    X = raw_data[:, 0:-1]
    return df, X, y, raw_data

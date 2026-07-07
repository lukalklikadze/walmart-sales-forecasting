import os
import pandas as pd
from src.config import DATA_DIR

def load_raw(data_dir: str = DATA_DIR):
    train    = pd.read_csv(os.path.join(data_dir, "train.csv"),    parse_dates=["Date"])
    test     = pd.read_csv(os.path.join(data_dir, "test.csv"),     parse_dates=["Date"])
    stores   = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    features = pd.read_csv(os.path.join(data_dir, "features.csv"), parse_dates=["Date"])

    features = features.drop(columns=["IsHoliday"])  # dup of the one in train/test

    def join(df):
        df = df.merge(stores,   on="Store",          how="left")
        df = df.merge(features, on=["Store", "Date"], how="left")
        return df

    return join(train), join(test), stores, features

import pandas as pd

def load_data_spondylo() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_spondylo.csv")
    return df


def load_data_panic_disorder() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_panic_disorder.csv")
    return df

def load_data_hernie() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_hernie.csv")
    return df

import pandas as pd

def load_data_spondylo() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_spondylo.csv")
    return df

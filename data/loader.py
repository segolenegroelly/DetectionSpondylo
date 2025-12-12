import pandas as pd

def load_data_spondylo() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_spondylo.csv")
    print(f"Nombre de lignes dupliquées : {df.duplicated().sum()}")
    return df.drop_duplicates()


def load_data_panic_disorder() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_panic_disorder.csv")
    print(f"Nombre de lignes dupliquées : {df.duplicated().sum()}")
    return df.drop_duplicates()

def load_data_hernie() -> pd.DataFrame:
    df = pd.read_csv("data/dataset_hernie.csv")
    print(f"Nombre de lignes dupliquées : {df.duplicated().sum()}")
    return df.drop_duplicates()

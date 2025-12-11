from typing import Tuple

import pandas as pd
import unicodedata
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split


class ModelDataset(torch.utils.data.Dataset):

    def __init__(self, input_ids: list, labels: list):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
        }

def tokenize(data:pd.DataFrame, token) -> pd.DataFrame :

    data['sequence'] = data['text_cleaned'].apply(
        lambda x: token.encode(
            x,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length'
        )
    )

    vocab_size = token.vocab_size
    seq_lengths = [len([t for t in seq if t != token.pad_token_id]) for seq in data['sequence']]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max(seq_lengths)}")
    print(f"Mean sequence length: {np.mean(seq_lengths):.1f}")
    return data

def cleaningData(data:pd.DataFrame) -> pd.DataFrame :
    data['text_cleaned']= [textCleaning(text) for text in data['symptome']]
    print(data[['symptome', 'text_cleaned']].head(10))
    return data


def textCleaning(text:str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r"[^a-z',\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def separateTrainTest(data:pd.DataFrame) -> Tuple[ModelDataset,ModelDataset]:
    train_df, test_df = train_test_split(
        data[['diagnostique','sequence']],
        test_size=0.2,
        random_state=42,
        stratify=data['diagnostique']
    )

    train_dataset = ModelDataset(
        input_ids=train_df['sequence'].tolist(),
        labels=train_df['diagnostique'].tolist(),
    )

    test_dataset = ModelDataset(
        input_ids=test_df['sequence'].tolist(),
        labels=test_df['diagnostique'].tolist(),
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset
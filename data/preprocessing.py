import pandas as pd
import unicodedata
import numpy as np
import re
from sklearn.model_selection import train_test_split


def tokenize(data:pd.DataFrame, token) -> pd.DataFrame :
    data['sequence'] = data['text_cleaned'].apply(lambda x: token.encode(x, add_special_tokens=True))
    vocab_size = token.vocab_size
    max_len = max(len(seq) for seq in data['sequence'])
    print(f"\n📊 Statistiques du vocabulaire :")
    print(f"  • Taille du vocabulaire : {vocab_size}")
    print(f"  • Longueur maximale : {max_len} mots")
    print(f"  • Longueur moyenne : {np.mean([len(s) for s in data['sequence']]):.1f} mots")

    print(data[['text_cleaned', 'sequence']].head(10))
    #On tokenize les donnes
    return data

def cleaningData(data:pd.DataFrame) -> pd.DataFrame :

    data['text_cleaned']= [textCleaning(text) for text in data['symptome']]
    print(data[['symptome', 'text_cleaned']].head(10))
    return data


def textCleaning(text:str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r"[^a-z',\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def separateTrainTest(data:pd.DataFrame) :
    return train_test_split(
        data[['diagnostique','sequence']],
        test_size=0.2,
        random_state=42,
        stratify=data['diagnostique']
    )
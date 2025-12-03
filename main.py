import streamlit as st
import fastapi
import torch

from config import MODEL_SPONDYLO_NAME, BASELINE_MODEL_NAME
from data.loader import load_data_spondylo
from data.preprocessing import cleaningData, tokenize, separateTrainTest
from model.baselineModel import getTokenizer, getBaselineModele
from model.retrainedModel import computeModel



st.set_page_config("Detection maladies")

token = getTokenizer()

#Génération d'un modèle pour la spondylolisthésis

dfSpondylo = load_data_spondylo()
dfSpondylo=cleaningData(dfSpondylo)
dfSpondylo=tokenize(dfSpondylo, token)
train_data, test_data = separateTrainTest(dfSpondylo)

model = computeModel(BASELINE_MODEL_NAME,train_data,test_data,MODEL_SPONDYLO_NAME)


#A SUPPRIMER
def predire_maladie(texte):
    inputs = token(texte, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    probabilite = torch.softmax(outputs.logits, dim=-1)[0][1].item()

    return {
        'a_un_rhume': bool(prediction.item()),
        'probabilite': probabilite
    }

texte_test = "Nose bleeding, headache, vommitting"
resultat = predire_maladie(texte_test)
print(f"Prédiction pour faux: {resultat}")

texte_test = "hip pain, loss of sensation, leg weakness"
resultat = predire_maladie(texte_test)
print(f"Prédiction pour vrai: {resultat}")
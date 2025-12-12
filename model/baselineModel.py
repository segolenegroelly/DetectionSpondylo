from transformers import AutoTokenizer, AutoModel

from config import BASELINE_MODEL_NAME, MODEL_SPONDYLO_NAME, MODEL_PANIC_NAME, MODEL_HERNIE_NAME
from data.loader import load_data_spondylo, load_data_hernie, load_data_panic_disorder
from data.preprocessing import cleaningData, separateTrainTest, tokenizeData
from model.retrainedModel import computeModel


def getTokenizer():
    return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def getBaselineModele():
    return AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#Génération d'un modèle pour la spondylolisthésis

def generateSpondyloModele(token):
    dfSpondylo = load_data_spondylo()
    dfSpondylo = cleaningData(dfSpondylo)
    dfSpondylo = tokenizeData(dfSpondylo, token)
    train_data, test_data = separateTrainTest(dfSpondylo)

    model = computeModel(BASELINE_MODEL_NAME, train_data, test_data, token, MODEL_SPONDYLO_NAME)
    return model,token

#Génération d'un modèle pour le trouble panic

def generatePanicDisorderModele(token):
    dfPanic = load_data_panic_disorder()
    dfPanic = cleaningData(dfPanic)
    dfPanic = tokenizeData(dfPanic, token)
    train_data, test_data = separateTrainTest(dfPanic)

    model = computeModel(BASELINE_MODEL_NAME, train_data, test_data, token, MODEL_PANIC_NAME)
    return model,token

#Génération d'un modèle pour le trouble panic

def generateHerniatedDiskModele(token):
    dfHernie = load_data_hernie()
    dfHernie = cleaningData(dfHernie)
    dfHernie = tokenizeData(dfHernie, token)
    train_data, test_data = separateTrainTest(dfHernie)

    model = computeModel(BASELINE_MODEL_NAME, train_data, test_data, token, MODEL_HERNIE_NAME)
    return model,token
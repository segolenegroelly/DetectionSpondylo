from transformers import AutoTokenizer, AutoModel


def getTokenizer():
    return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def getBaselineModele():
    return AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
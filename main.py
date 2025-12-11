from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import MODEL_SPONDYLO_NAME, MODEL_PANIC_NAME, MODEL_HERNIE_NAME, GENERATE_MODEL
from model.baselineModel import getTokenizer, generateSpondyloModele, generatePanicDisorderModele, \
    generateHerniatedDiskModele
from model.modelUse import generatePrediction, generateExemple
from model.retrainedModel import loadSavedModelAndToken

if(GENERATE_MODEL):
    token = getTokenizer()
    modeleSpondylo,tokenSpondylo = generateSpondyloModele(token)
    modelePanic, tokenPanic = generatePanicDisorderModele(token)
    modeleHernie, tokenHernie = generateHerniatedDiskModele(token)
else:
    modeleSpondylo,tokenSpondylo = loadSavedModelAndToken(MODEL_SPONDYLO_NAME)
    modelePanic, tokenPanic = loadSavedModelAndToken(MODEL_PANIC_NAME)
    modeleHernie, tokenHernie = loadSavedModelAndToken(MODEL_HERNIE_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TexteInput(BaseModel):
    texte: str

@app.post("/detection")
async def detection(data: TexteInput):
    resultat = generatePrediction(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic, modeleHernie, tokenHernie,data.texte)
    return {
        "resultat": resultat
    }

@app.get("/exemple")
async def exemple():
    resultat = generateExemple(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic, modeleHernie, tokenHernie)
    return {
        "resultat": resultat
    }

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
from typing import Tuple

import pandas as pd
import torch


def generationReponse(df: pd.DataFrame) -> str:
    result = '''<div class="container">
        <h2>Determination de pathologie</h2>
        <table>
            <thead>
                <tr>
                    <th>Symptômes</th>
                    <th>Maladie</th>
                    <th>Spondylolisthésis</th>
                    <th>Taux de confiance SP</th>
                    <th>Trouble anxieux panique</th>
                    <th>Taux de confiance TA</th>
                    <th>Hernie discale</th>
                    <th>Taux de confiance HD</th>
                </tr>
            </thead>
            <tbody>'''
    for index, row in df.iterrows():
        result += f'''<tr><td>{row['symptomes']}</td>
        <td>{row['maladie']}</td>
        <td>{row['spondylo']}</td>
        <td>{row['tc_spondylo']}</td>
        <td>{row['panic']}</td>
        <td>{row['tc_panic']}</td>
        <td>{row['hernie']}</td>
        <td>{row['tc_hernie']}</td>
        </tr>'''
    result += '''</tbody>
            </table>
        </div>'''
    return result


def generateExemple(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic, modeleHernie, tokenHernie) -> str:
    texte_test = pd.DataFrame({
        'symptomes': ["hip pain, loss of sensation, leg weakness",
                      "shortness of breath, insomnia, abnormal involuntary movements, chest tightness, palpitations",
                      "arm pain, neck pain, back pain, loss of sensation, paresthesia, arm weakness",
                      "abnormal appearing skin, foot or toe swelling, hand or finger lump or mass, skin on leg or foot looks infected, sinus congestion, skin on arm or hand looks infected",
                      "sharp abdominal pain, vomiting, diarrhea, heartburn, fever, chills",
                      "suprapubic pain, sharp abdominal pain, vomiting, painful urination, lower abdominal pain, intermenstrual bleeding, burning abdominal pain",
                      "diminished vision, pain in eye, eye redness, lacrimation",
                      "wrist pain, wrist swelling, ankle pain, problems with movement, knee stiffness or tightness, ankle swelling"],
        'maladie': ["spondylolisthesis",
                    "panic disorder",
                    "herniated disk",
                    "paronychia",
                    "white blood cell disease",
                    "pelvic inflammatory disease",
                    "conjunctivitis due to virus",
                    "joint effusion"]
    })

    df = pd.DataFrame()
    i = 1
    for index, row in texte_test.iterrows():
        df = pd.concat([df, generationTableauPrediction(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic,
                                                        modeleHernie, tokenHernie, row['symptomes'], row['maladie'],
                                                        i)], ignore_index=True)
        i = i + 1
    return generationReponse(df)


def generatePrediction(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic, modeleHernie, tokenHernie,
                       symptomes: str) -> str:
    df = generationTableauPrediction(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic, modeleHernie, tokenHernie,
                                     symptomes, '', 1)
    return generationReponse(df)


def predireMaladie(model, token, texte)->tuple[str,float]:
    inputs = token(texte, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilite = torch.softmax(outputs.logits)[0][1].item()
    return "oui" if probabilite>0.40 else "non",probabilite



def generationTableauPrediction(modeleSpondylo, tokenSpondylo, modelePanic, tokenPanic, modeleHernie, tokenHernie,
                                symptomes: str, maladie: str, index: int) -> pd.DataFrame:
    predSp, probaSp = predireMaladie(modeleSpondylo, tokenSpondylo, symptomes)
    predPd, probaPd = predireMaladie(modelePanic, tokenPanic, symptomes)
    predHd, probaHd = predireMaladie(modeleHernie, tokenHernie, symptomes)
    df = pd.DataFrame({
        'symptomes': symptomes,
        'maladie': maladie,
        'spondylo': predSp,
        'tc_spondylo': round(probaSp,2),
        'panic': predPd,
        'tc_panic': round(probaPd,2),
        'hernie': predHd,
        'tc_hernie': round(probaHd,2)
    }, index=[index])
    return df

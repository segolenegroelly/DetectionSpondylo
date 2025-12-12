
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
        'symptomes': ["low back pain, problems with movement, paresthesia, leg cramps or spasms, leg weakness",
            "hip pain, back pain, neck pain, low back pain, problems with movement, loss of sensation, leg cramps or spasms",
            "leg pain, back pain, neck pain, problems with movement, loss of sensation, paresthesia, leg cramps or spasms",
            "leg pain, hip pain, neck pain, problems with movement, paresthesia, leg weakness",
            "arm pain, back pain, neck pain, paresthesia, shoulder pain, arm weakness",
            "hip pain, back pain, low back pain, loss of sensation, paresthesia, shoulder pain, leg weakness",
            "herniated disk	loss of sensation, paresthesia, shoulder pain",
            "herniated disk	hip pain, arm pain, back pain, low back pain, loss of sensation, paresthesia, leg weakness",
            "depressive or psychotic symptoms, irregular heartbeat, breathing fast",
            "insomnia, palpitations, irregular heartbeat",
            "depression, shortness of breath, depressive or psychotic symptoms, dizziness, insomnia, abnormal involuntary movements, irregular heartbeat",
            "depression, depressive or psychotic symptoms, insomnia, abnormal involuntary movements, chest tightness, palpitations"],
        'maladie': ["spondylolisthesis",
            "spondylolisthesis",
            "spondylolisthesis",
            "spondylolisthesis",
            "herniated disk",
            "herniated disk",
            "herniated disk",
            "herniated disk",
            "panic disorder",
            "panic disorder",
            "panic disorder",
            "panic disorder"]
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


def predireMaladie(model, token, texte:str)->tuple[str,float]:
    inputs = token(texte, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilite = torch.sigmoid(outputs.logits).item()
    return "oui" if probabilite>0.50 else "non",probabilite

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

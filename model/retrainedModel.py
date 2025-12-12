from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction, AutoTokenizer, \
    EarlyStoppingCallback
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from scipy.special import expit


def computeModel(baselineModelName:str,train_dataset, test_dataset,token,nom:str)->AutoModelForSequenceClassification:
    model = generateModel(baselineModelName)
    trainer = prepareTrainer(model, train_dataset, test_dataset)
    training(trainer)
    saveModelAndTokenizer(trainer,token,nom)
    return model

def generateModel(baselineModelName:str)-> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(
        baselineModelName,
        num_labels=1
    )

def prepareTrainer(model, train_dataset, test_dataset)->Trainer:
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc_pr",
        greater_is_better=True
    )

    # Pour test en interne, retirer sur lightning
    training_args.no_cuda = True
    training_args.pin_memory =False

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

def compute_metrics(pred: EvalPrediction):
    predictions, labels = pred
    predictions = predictions.flatten()

    predictions_prob = expit(predictions)

    pred_binary = (predictions_prob > 0.5).astype(int)
    labels_binary = (labels > 0.5).astype(int)

    # Métriques de classification
    precision = precision_score(labels_binary, pred_binary, zero_division=0)
    recall = recall_score(labels_binary, pred_binary, zero_division=0)
    f1 = f1_score(labels_binary, pred_binary, zero_division=0)

    # Métriques basées sur les probabilités
    auc_roc = roc_auc_score(labels_binary, predictions_prob)
    auc_pr = average_precision_score(labels_binary, predictions_prob)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }

def training(trainer:Trainer):
    print(f"Début de l'entrainement")
    trainer.train()
    results = trainer.evaluate()
    print(f"Résultats : {results}")

def saveModelAndTokenizer(trainer,token,nom:str):
    trainer.save_model(f'./{nom}Model')
    print(f"Modèle sauvegardé")
    token.save_pretrained(f'./{nom}Tokenizer')
    print(f"Tokenizer sauvegardé")

def loadSavedModelAndToken(nom:str)->tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(f'./{nom}Model')
    tokenizer = AutoTokenizer.from_pretrained(f'./{nom}Tokenizer')
    return model,tokenizer
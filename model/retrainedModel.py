from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction, AutoTokenizer, \
    EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def computeModel(baselineModelName:str,train_dataset, test_dataset,token,nom:str)->AutoModelForSequenceClassification:
    model = generateModel(baselineModelName)
    trainer = prepareTrainer(model, train_dataset, test_dataset)
    training(trainer)
    saveModelAndTokenizer(model,token,nom)
    return model

def generateModel(baselineModelName:str)-> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(
        baselineModelName,
        num_labels=2
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
        metric_for_best_model="eval_recall",
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
def compute_metrics(pred:EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def training(trainer:Trainer):
    print(f"Début de l'entrainement")
    trainer.train()
    results = trainer.evaluate()
    print(f"Résultats : {results}")

def saveModelAndTokenizer(model,token,nom:str):
    model.save_pretrained(f'./{nom}Model')
    print(f"Modèle sauvegardé")
    token.save_pretrained(f'./{nom}Tokenizer')
    print(f"Tokenizer sauvegardé")

def loadSavedModelAndToken(nom:str)->tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(f'./{nom}Model')
    tokenizer = AutoTokenizer.from_pretrained(f'./{nom}Tokenizer')
    return model,tokenizer
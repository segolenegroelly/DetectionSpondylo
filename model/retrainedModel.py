from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def computeModel(baselineModelName:str,train_dataset, test_dataset,nom:str):
    model = generateModel(baselineModelName)
    trainer = prepareTrainer(model, train_dataset, test_dataset)
    training(trainer)
    saveModelAndTokenizer(model,nom)
    return model

def generateModel(baselineModelName:str):
    return AutoModelForSequenceClassification.from_pretrained(
        baselineModelName,
        num_labels=2
    )

def prepareTrainer(model, train_dataset, test_dataset)->Trainer:
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
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
    token.save_pretrained(f'./{nom}Tokenizer')
import evaluate
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import gc

from datasets import Dataset
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    accuracy_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    Trainer
)
from transformers.training_args import TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data"
train_data_path = os.path.join(data_dir, 'train_data_llava.json')
test_data_path = os.path.join(data_dir, 'test_data_llava.json')
models_dir = "models"

def generate_df_from_json(json_path):
    with open(json_path) as f:
        data: dict = json.load(f)
        data_formated = {'caption': [], 'label': []}
        for value in data.values():
            data_formated['caption'].append(value['caption'])
            data_formated['label'].append(value['label'])
  
    df = pd.DataFrame.from_dict(data_formated)
    return df

id2label = {
    0: 'Cultural_Religious',
    1: 'Fauna_Flora',
    2: 'Gastronomy',
    3: 'Nature',
    4: 'Sports',
    5: 'Urban_Rural'
}

label2id = {v: k for k, v in id2label.items()}

train_df = generate_df_from_json(train_data_path)
test_df = generate_df_from_json(test_data_path)
train_df = pd.concat([train_df, test_df], axis=0)

train_data = Dataset.from_pandas(train_df)

label_count = train_df.groupby(by='label').count()
label_count.columns = ['count']
labels = [id2label[id] for id in label_count.index]

model_ids = [
    "google-bert/bert-base-uncased",
    "distilbert/distilbert-base-uncased",
    "FacebookAI/roberta-base"
]

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["caption"], truncation=True)

data_collator = None

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy_result = accuracy_metric.compute(predictions=predictions, references=labels)
    precision_result = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    recall_result = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    
    return {
        'accuracy': accuracy_result['accuracy'],
        'precision': precision_result['precision'],
        'recall': recall_result['recall']
    }

# Results dictionary
results = {}

# K-Fold parameters
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate over each model
for model_id in model_ids:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    tokenized_train_data = train_data.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    train_labels = tokenized_train_data['label']
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_per_class = {cls: [] for cls in label2id.keys()}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(tokenized_train_data, train_labels)):
        # Create train and validation sets for this fold
        train_subset = tokenized_train_data.select(train_idx.tolist())
        val_subset = tokenized_train_data.select(val_idx.tolist())

        # Initialize model for the current fold
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=6, id2label=id2label, label2id=label2id
        )

        # Define training arguments for this fold
        training_args = TrainingArguments(
            output_dir=f"{models_dir}/{model_id.split('/')[-1]}-fold-{fold+1}",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        # Initialize Trainer for this fold
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model on the validation subset
        eval_results = trainer.evaluate(eval_dataset=val_subset)
        
        predictions, labels, _ = trainer.predict(val_subset)
        predictions = np.argmax(predictions, axis=1)
        
        # Compute the confusion matrix
        cm = confusion_matrix(labels, predictions, labels=list(label2id.values()))

        # Store metrics for this fold
        fold_accuracies.append(eval_results['eval_accuracy'])
        fold_precisions.append(eval_results['eval_precision'])
        fold_recalls.append(eval_results['eval_recall'])

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        classes = []
        class_accuracy = []
        
        for id, cls in id2label.items():
            fold_per_class[cls].append(cm.diagonal()[id])

        # Delete the model and free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute mean metrics across all folds
    mean_accuracy = np.mean(fold_accuracies)
    mean_precision = np.mean(fold_precisions)
    mean_recall = np.mean(fold_recalls)
    mean_per_class = {cls: np.mean(fold_per_class[cls]) for cls in fold_per_class.keys()}
    
    # Transform all means into % with 2 decimals
    mean_accuracy = round(mean_accuracy*100, 2)
    mean_precision = round(mean_precision*100, 2)
    mean_recall = round(mean_recall*100, 2)
    mean_per_class = {cls: round(acc*100, 2) for cls, acc in mean_per_class.items()}
    
    # Print results for this model
    print(f"Model: {model_id}")
    print(f"Mean Accuracy: {mean_accuracy}%")
    print(f"Mean Precision: {mean_precision}%")
    print(f"Mean Recall: {mean_recall}%")
    print(f"Mean Per Class Accuracy: {mean_per_class}")
    print("--------------------------------------------------")

    # Save results for this model
    results[model_id] = {
        'mean_accuracy': mean_accuracy,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall
    }

# Save results to a JSON file
with open(os.path.join(models_dir, 'model_results_kfold.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Save per class result into csv file
df = pd.DataFrame.from_dict(mean_per_class, orient='index', columns=['Accuracy'])
df.index.name = 'Class'
df.to_csv(os.path.join(models_dir, 'per_class_accuracy.csv'))
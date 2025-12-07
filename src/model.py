import json
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch

def load_data(file_path="punjabi_songs_with_mood.json"):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna(subset=['lyrics', 'mood'])
    df['lyrics'] = df['lyrics'].astype(str)
    df['mood'] = df['mood'].astype(str)
    print(f"Loaded {len(df)} samples")
    print(f"Mood distribution:\n{df['mood'].value_counts()}")
    return df

def prepare_dataset(df):
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['mood'])
    label_names = list(label_encoder.classes_)
    features = Features({
        'lyrics': Value(dtype='string'),
        'labels': ClassLabel(names=label_names)
    })
    dataset = Dataset.from_pandas(df[['lyrics', 'labels']], features=features)
    train_test = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='labels')
    print(f"Dataset splits: Train = {len(train_test['train'])}, Test = {len(train_test['test'])}")
    return train_test, label_encoder

def tokenize_dataset(dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(
            examples["lyrics"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization completed")
    return tokenized_dataset, tokenizer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_model(tokenized_dataset, tokenizer, label_encoder, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_total_limit=2,
        seed=42,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    print(f"Training for {training_args.num_train_epochs} epochs...")
    trainer.train()
    print("Final evaluation:")
    final_metrics = trainer.evaluate()
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    return trainer

def save_artifacts(trainer, tokenizer, label_encoder, output_dir):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(f"{output_dir}/label_mapping.json", "w", encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    print("Model and artifacts saved!")

def main():
    DATA_FILE = 'punjabi_songs_with_mood.json'
    MODEL_NAME = "ai4bharat/indic-bert"
    OUTPUT_DIR = "./indic_bert_punjabi_mood"

    print(f"Punjabi Mood Classifier using {MODEL_NAME}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    df = load_data(DATA_FILE)
    dataset, label_encoder = prepare_dataset(df)
    tokenized_dataset, tokenizer = tokenize_dataset(dataset, MODEL_NAME)
    trainer = train_model(tokenized_dataset, tokenizer, label_encoder, MODEL_NAME, OUTPUT_DIR)
    save_artifacts(trainer, tokenizer, label_encoder, OUTPUT_DIR)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

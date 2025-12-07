import torch
import json
import pandas as pd
import numpy as np
import re
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import torch.nn as nn
import os
from collections import Counter

def enhanced_preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[!?]{2,}', lambda x: x.group()[0], text)
    
    # Punjabi-specific cleaning
    text = re.sub(r'[^\u0A00-\u0A7F\s]', '', text)  # Keep only Punjabi Unicode chars
    
    # Remove excessive punctuation but keep some for context
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{2,}', '--', text)
    
    # Remove extra whitespace around punctuation
    text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)
    
    # Remove very short words but keep important Punjabi characters
    words = text.split()
    words = [word for word in words if len(word) >= 2 or word in ['ਆ', 'ਏ', 'ਓ', 'ੳ', 'ਅ']]
    
    return ' '.join(words).strip()

def augment_training_data(train_dataset):
    augmented_data = []
    for example in train_dataset:
        # Add slight variations
        text = example['lyrics']
        # Remove random words (20% chance)
        if random.random() < 0.2:
            words = text.split()
            if len(words) > 5:
                remove_idx = random.randint(0, len(words)-1)
                words.pop(remove_idx)
                augmented_data.append({
                    'lyrics': ' '.join(words),
                    'labels': example['labels']
                })
    
    # Add augmented examples to training data
    train_dataset = train_dataset.add_items(augmented_data)
    return train_dataset

def analyze_dataset_quality(df):
    print("\n" + "="*50)
    print("DATASET QUALITY ANALYSIS")
    print("="*50)
    
    class_counts = df['mood'].value_counts()
    print(f"\nClass Distribution:")
    for mood, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {mood}: {count} samples ({percentage:.1f}%)")
    
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("  ⚠️  HIGH IMBALANCE DETECTED - This may hurt performance")
    
    df['text_length'] = df['lyrics'].str.len()
    df['word_count'] = df['lyrics'].str.split().str.len()
    
    print(f"\nText Statistics:")
    print(f"  Average character length: {df['text_length'].mean():.1f}")
    print(f"  Average word count: {df['word_count'].mean():.1f}")
    print(f"  Min words: {df['word_count'].min()}")
    print(f"  Max words: {df['word_count'].max()}")
    
    short_texts = df[df['word_count'] < 10]
    if len(short_texts) > 0:
        print(f"  ⚠️  {len(short_texts)} samples have < 10 words (might lack context)")
    
    duplicates = df.duplicated(subset=['lyrics']).sum()
    if duplicates > 0:
        print(f"  ⚠️  {duplicates} duplicate lyrics found")
    
    return {
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio,
        'avg_length': df['text_length'].mean(),
        'avg_words': df['word_count'].mean()
    }

def load_and_clean_data(file_path="punjabi_songs_with_mood.json", min_samples_per_class=20):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Initial dataset size: {len(df)}")
    
    df = df.dropna(subset=['lyrics', 'mood'])
    print(f"After removing missing values: {len(df)}")
    
    df['lyrics'] = df['lyrics'].apply(enhanced_preprocess_text)
    
    df = df[df['lyrics'].str.strip() != '']
    df = df[df['lyrics'].str.split().str.len() >= 8]
    print(f"After text filtering: {len(df)}")
    
    class_counts = df['mood'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df['mood'].isin(valid_classes)]
    print(f"After removing small classes: {len(df)}")
    
    df = df.drop_duplicates(subset=['lyrics'])
    print(f"After removing duplicates: {len(df)}")
    
    analysis = analyze_dataset_quality(df)
    
    return df, analysis

def prepare_enhanced_dataset(df, test_size=0.2, val_size=0.1):
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['mood'])
    
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['labels'], random_state=42
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, stratify=train_val_df['labels'], random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    
    train_dataset = Dataset.from_pandas(train_df[['lyrics', 'labels']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['lyrics', 'labels']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['lyrics', 'labels']].reset_index(drop=True))
    
    # Apply data augmentation to training set
    train_dataset = augment_training_data(train_dataset)
    
    return {
        'train': train_dataset, 
        'validation': val_dataset,
        'test': test_dataset
    }, label_encoder

def tokenize_dataset_enhanced(dataset, model_name, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["lyrics"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=True
        )
    
    tokenized = {k: v.map(tokenize_fn, batched=True) for k, v in dataset.items()}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return tokenized, tokenizer, data_collator

class ImprovedWeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if self.class_weights is not None:
            weight = self.class_weights.to(outputs.logits.device)
        else:
            weight = None
            
        loss_fct = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=0.1
        )
        
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence_penalty = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        loss = loss + 0.01 * confidence_penalty
        
        return (loss, outputs) if return_outputs else loss

def train_improved_model(tokenized_dataset, tokenizer, data_collator, label_encoder, model_name, output_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.3,
        ignore_mismatched_sizes=True
    )
    
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(tokenized_dataset["train"]["labels"]),
        y=tokenized_dataset["train"]["labels"]
    )
    
    class_weights = torch.FloatTensor(raw_weights)
    class_weights = torch.clamp(class_weights, min=0.5, max=2.5)
    
    print("Smoothed class weights:", class_weights.numpy())
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps", 
        save_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        lr_scheduler_type='cosine_with_restarts',
        optim="adamw_torch",
        max_grad_norm=0.5,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.98,
        group_by_length=True,
        remove_unused_columns=False,
    )
    
    trainer = ImprovedWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
            "f1": precision_recall_fscore_support(
                p.label_ids, np.argmax(p.predictions, axis=1), average='weighted')[2]
        },
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=8,
                early_stopping_threshold=0.001
            )
        ],
        class_weights=class_weights
    )
    
    print("\nStarting enhanced training...")
    trainer.train()
    
    history = trainer.state.log_history
    train_losses = [x['loss'] for x in history if 'loss' in x]
    eval_losses = [x['eval_loss'] for x in history if 'eval_loss' in x]
    
    print("\nTraining Summary:")
    if train_losses and eval_losses:
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Eval Loss: {eval_losses[-1]:.4f}")
        print(f"Loss Gap: {abs(train_losses[-1] - eval_losses[-1]):.4f}")
        
        if abs(train_losses[-1] - eval_losses[-1]) > 0.3:
            print("⚠️  Large train/eval loss gap suggests overfitting")
    
    return trainer

def comprehensive_evaluation(trainer, tokenized_dataset, label_encoder):
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    val_predictions = trainer.predict(tokenized_dataset["validation"])
    val_pred = np.argmax(val_predictions.predictions, axis=1)
    val_true = val_predictions.label_ids
    
    if "test" in tokenized_dataset:
        test_predictions = trainer.predict(tokenized_dataset["test"])
        test_pred = np.argmax(test_predictions.predictions, axis=1)
        test_true = test_predictions.label_ids
        
        test_accuracy = accuracy_score(test_true, test_pred)
        print(f"TEST SET ACCURACY: {test_accuracy:.4f} ({test_accuracy:.2%})")
    
    val_accuracy = accuracy_score(val_true, val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_true, val_pred, average='weighted')
    
    print(f"\nVALIDATION PERFORMANCE:")
    print(f"Accuracy:  {val_accuracy:.4f} ({val_accuracy:.2%})")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    
    print(f"\nPER-CLASS DETAILED REPORT:")
    class_report = classification_report(
        val_true, val_pred, 
        target_names=label_encoder.classes_, 
        digits=4,
        zero_division=0
    )
    print(class_report)
    
    cm = confusion_matrix(val_true, val_pred)
    print(f"\nCONFUSION MATRIX:")
    print("Predicted →")
    print("Actual ↓")
    print("         ", end="")
    for class_name in label_encoder.classes_:
        print(f"{class_name[:8]:>8}", end=" ")
    print()
    
    for i, actual_class in enumerate(label_encoder.classes_):
        print(f"{actual_class[:8]:>8} | ", end="")
        for j in range(len(label_encoder.classes_)):
            print(f"{cm[i][j]:>6}", end=" ")
        print()
    
    errors = []
    for i in range(len(label_encoder.classes_)):
        for j in range(len(label_encoder.classes_)):
            if i != j and cm[i][j] > 0:
                actual = label_encoder.classes_[i]
                predicted = label_encoder.classes_[j]
                count = cm[i][j]
                errors.append((actual, predicted, count))
    
    errors.sort(key=lambda x: x[2], reverse=True)
    print("\nERROR ANALYSIS:")
    print("Most common misclassifications:")
    for actual, predicted, count in errors[:5]:
        print(f"  {actual} → {predicted}: {count} errors")
    
    print(f"\n" + "="*70)
    if val_accuracy >= 0.80:
        print(f"🎉 SUCCESS: {val_accuracy:.2%} accuracy achieved!")
    elif val_accuracy >= 0.75:
        print(f"📈 GOOD: {val_accuracy:.2%} accuracy - Close to target")
        print("Suggestions: Increase epochs to 30, try learning_rate=5e-6")
    elif val_accuracy >= 0.65:
        print(f"📊 MODERATE: {val_accuracy:.2%} accuracy")
        print("Suggestions: Check for data quality issues, consider data augmentation")
    else:
        print(f"📉 LOW: {val_accuracy:.2%} accuracy")
        print("Suggestions: Review data quality, class definitions, and preprocessing")
    print("="*70)
    
    return {
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'test_accuracy': test_accuracy if "test" in tokenized_dataset else None,
        'confusion_matrix': cm,
        'errors': errors
    }

def save_model(trainer, tokenizer, label_encoder, output_dir, evaluation_results=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    print("Saving label mappings...")
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        label_mapping = {
            "id2label": {i: label for i, label in enumerate(label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(label_encoder.classes_)}
        }
        json.dump(label_mapping, f, indent=2)
        print(f"Label mapping: {label_mapping['id2label']}")
    
    if evaluation_results:
        print("Saving evaluation results...")
        with open(f"{output_dir}/evaluation_results.json", "w") as f:
            results_to_save = {
                'val_accuracy': float(evaluation_results['val_accuracy']),
                'val_precision': float(evaluation_results['val_precision']),
                'val_recall': float(evaluation_results['val_recall']),
                'val_f1': float(evaluation_results['val_f1']),
                'test_accuracy': float(evaluation_results['test_accuracy']) if evaluation_results['test_accuracy'] is not None else None,
                'class_names': label_encoder.classes_.tolist(),
                'target_accuracy': 0.80,
                'achieved_target': evaluation_results['val_accuracy'] >= 0.80
            }
            json.dump(results_to_save, f, indent=2)
    
    print(f"\nAll files saved successfully to: {output_dir}")
    print("Files created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

def main():
    DATA_FILE = "punjabi_songs_with_mood.json"
    MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
    OUTPUT_DIR = "./improved_punjabi_mood_model"
    
    print("🚀 ENHANCED PUNJABI MOOD CLASSIFICATION PIPELINE")
    print("="*70)
    
    try:
        print("Step 1: Loading and analyzing data...")
        df, data_analysis = load_and_clean_data(DATA_FILE)
        
        if len(df) < 100:
            print("⚠️  Warning: Very small dataset. Consider collecting more data.")
        
        print("\nStep 2: Preparing enhanced dataset...")
        dataset, label_encoder = prepare_enhanced_dataset(df)
        
        print("\nStep 3: Enhanced tokenization...")
        tokenized_dataset, tokenizer, data_collator = tokenize_dataset_enhanced(dataset, MODEL_NAME)
        
        print("\nStep 4: Training improved model...")
        trainer = train_improved_model(
            tokenized_dataset, tokenizer, data_collator, 
            label_encoder, MODEL_NAME, OUTPUT_DIR
        )
        
        print("\nStep 5: Comprehensive evaluation...")
        results = comprehensive_evaluation(trainer, tokenized_dataset, label_encoder)
        
        print("\nStep 6: Saving improved model...")
        save_model(trainer, tokenizer, label_encoder, OUTPUT_DIR, results)
        
        print(f"\n🎯 FINAL RESULT: {results['val_accuracy']:.2%} validation accuracy")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
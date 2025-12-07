# Patch torch.serialization.load to ignore RNG loading errors
import torch.serialization

_original_load = torch.serialization.load

def safe_load(f, *args, **kwargs):
    try:
        return _original_load(f, *args, **kwargs)
    except AttributeError as e:
        if "'str' object has no attribute '__module__'" in str(e):
            print("[Warning] Ignored RNG state loading error during checkpoint load")
            return None
        raise e

torch.serialization.load = safe_load

import json
import pandas as pd
import numpy as np
import pickle
import os
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    TrainerCallback
)
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

# Try to import indic_transliteration for better text processing
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    print("[Warning] indic_transliteration not available. Install with: pip install indic-transliteration")
    TRANSLITERATION_AVAILABLE = False

def preprocess_text(text):
    """Enhanced text preprocessing for better mood classification"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Normalize Gurmukhi text if transliteration is available
    if TRANSLITERATION_AVAILABLE:
        try:
            # Check if text contains Gurmukhi characters
            if any(ord(char) >= 0x0A00 and ord(char) <= 0x0A7F for char in text):
                # Convert to Roman for better processing
                romanized = transliterate(text, sanscript.GURMUKHI, sanscript.ITRANS)
                return romanized
        except Exception:
            pass
    
    return text

def load_data(file_path="punjabi_songs_with_mood.json"):
    """Load and prepare data with enhanced preprocessing"""
    print(f"Loading data from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if 'lyrics' not in df.columns or 'mood' not in df.columns:
        raise ValueError("Required columns 'lyrics' and 'mood' not found")

    # Initial cleaning
    df = df.dropna(subset=['lyrics', 'mood'])
    df['lyrics'] = df['lyrics'].astype(str)
    df['mood'] = df['mood'].astype(str)
    
    # Enhanced text preprocessing
    print("Preprocessing text...")
    df['lyrics'] = df['lyrics'].apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df['lyrics'].str.strip() != '']
    
    # Remove very short texts (less than 3 words)
    df = df[df['lyrics'].str.split().str.len() >= 3]

    print(f"Loaded {len(df)} samples after preprocessing")
    print(f"Mood distribution:\n{df['mood'].value_counts()}")
    
    # Check for class imbalance
    mood_counts = df['mood'].value_counts()
    min_samples = mood_counts.min()
    max_samples = mood_counts.max()
    imbalance_ratio = max_samples / min_samples
    
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("[Warning] Significant class imbalance detected!")

    return df

def augment_data(df, augmentation_factor=0.3):
    """Conservative data augmentation to prevent overfitting"""
    print("Applying conservative data augmentation...")
    
    augmented_data = []
    
    for _, row in df.iterrows():
        text = row['lyrics']
        mood = row['mood']
        
        # Original text
        augmented_data.append({'lyrics': text, 'mood': mood})
        
        # Light augmentation only for minority classes
        mood_count = (df['mood'] == mood).sum()
        if mood_count < df['mood'].value_counts().mean() and np.random.random() < augmentation_factor:
            # Simple word order changes for longer texts
            words = text.split()
            if len(words) > 5:
                # Shuffle middle portion slightly
                mid_start = len(words) // 4
                mid_end = 3 * len(words) // 4
                middle_words = words[mid_start:mid_end]
                np.random.shuffle(middle_words)
                augmented_text = ' '.join(words[:mid_start] + middle_words + words[mid_end:])
                augmented_data.append({'lyrics': augmented_text, 'mood': mood})
    
    augmented_df = pd.DataFrame(augmented_data)
    print(f"Augmented dataset size: {len(augmented_df)} (original: {len(df)})")
    
    return augmented_df

def prepare_dataset(df, use_augmentation=True):
    """Prepare dataset with balanced splits"""
    print("Preparing dataset...")
    
    # Apply conservative data augmentation if enabled
    if use_augmentation:
        df = augment_data(df, augmentation_factor=0.2)  # Reduced augmentation
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['mood'])

    print("Label mapping:")
    label_names = list(label_encoder.classes_)
    for i, mood in enumerate(label_names):
        count = (df['labels'] == i).sum()
        print(f"  {mood}: {i} ({count} samples)")

    # Create features with proper schema
    features = Features({
        'lyrics': Value(dtype='string'),
        'labels': ClassLabel(names=label_names)
    })

    df_reset = df[['lyrics', 'labels']].reset_index(drop=True)
    dataset = Dataset.from_pandas(df_reset, features=features)

    # Stratified split with proper validation set
    train_test = dataset.train_test_split(test_size=0.25, seed=42, stratify_by_column='labels')
    train_val = train_test['train'].train_test_split(test_size=0.25, seed=42, stratify_by_column='labels')
    
    final_dataset = {
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_test['test']
    }

    print("Dataset splits:")
    print(f"  Train: {len(final_dataset['train'])} samples")
    print(f"  Validation: {len(final_dataset['validation'])} samples")
    print(f"  Test: {len(final_dataset['test'])} samples")

    return final_dataset, label_encoder

def tokenize_dataset(dataset, model_name):
    """Optimized tokenization with consistent max length"""
    print(f"Tokenizing data with {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Conservative max length to prevent overfitting on long sequences
    MAX_LENGTH = 256  # Reduced from 512

    def tokenize_function(examples):
        return tokenizer(
            examples["lyrics"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False
        )

    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(tokenize_function, batched=True)
    
    print(f"Tokenization completed with max_length={MAX_LENGTH}")
    return tokenized_dataset, tokenizer

def compute_enhanced_metrics(eval_pred):
    """Comprehensive evaluation metrics"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    # Basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    # Macro averages
    macro_f1 = np.mean(f1_per_class)
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "f1_macro": macro_f1,
        "precision_weighted": precision,
        "precision_macro": macro_precision,
        "recall_weighted": recall,
        "recall_macro": macro_recall,
    }

class ProgressTrackingCallback(TrainerCallback):
    """Custom callback for detailed progress tracking and saving"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.progress_file = f"{output_dir}/training_progress.json"
        self.progress_data = {
            "training_history": [],
            "best_metrics": {},
            "epoch_summaries": []
        }
        os.makedirs(output_dir, exist_ok=True)
    
    def on_epoch_end(self, args, state, control, model, tokenizer, eval_dataloader=None, **kwargs):
        """Called at the end of each epoch"""
        current_epoch = int(state.epoch)
        
        # Get the latest log entry
        if state.log_history:
            latest_log = state.log_history[-1]
            
            epoch_summary = {
                "epoch": current_epoch,
                "timestamp": datetime.now().isoformat(),
                "train_loss": latest_log.get("train_loss", "N/A"),
                "eval_loss": latest_log.get("eval_loss", "N/A"),
                "eval_accuracy": latest_log.get("eval_accuracy", "N/A"),
                "eval_f1_weighted": latest_log.get("eval_f1_weighted", "N/A"),
                "eval_f1_macro": latest_log.get("eval_f1_macro", "N/A"),
                "learning_rate": latest_log.get("learning_rate", "N/A"),
                "step": state.global_step
            }
            
            self.progress_data["epoch_summaries"].append(epoch_summary)
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"EPOCH {current_epoch} COMPLETED")
            print(f"{'='*60}")
            print(f"Step: {state.global_step}")
            print(f"Train Loss: {epoch_summary['train_loss']:.4f}" if isinstance(epoch_summary['train_loss'], float) else f"Train Loss: {epoch_summary['train_loss']}")
            print(f"Eval Loss:  {epoch_summary['eval_loss']:.4f}" if isinstance(epoch_summary['eval_loss'], float) else f"Eval Loss: {epoch_summary['eval_loss']}")
            print(f"Accuracy:   {epoch_summary['eval_accuracy']:.4f}" if isinstance(epoch_summary['eval_accuracy'], float) else f"Accuracy: {epoch_summary['eval_accuracy']}")
            print(f"F1 Score:   {epoch_summary['eval_f1_weighted']:.4f}" if isinstance(epoch_summary['eval_f1_weighted'], float) else f"F1 Score: {epoch_summary['eval_f1_weighted']}")
            
            # Check if this is the best model so far
            if isinstance(epoch_summary['eval_loss'], float):
                if not self.progress_data["best_metrics"] or epoch_summary['eval_loss'] < self.progress_data["best_metrics"].get("best_eval_loss", float('inf')):
                    self.progress_data["best_metrics"] = {
                        "best_epoch": current_epoch,
                        "best_eval_loss": epoch_summary['eval_loss'],
                        "best_accuracy": epoch_summary['eval_accuracy'],
                        "best_f1": epoch_summary['eval_f1_weighted'],
                        "timestamp": epoch_summary['timestamp']
                    }
                    print(f"🎉 NEW BEST MODEL! (Eval Loss: {epoch_summary['eval_loss']:.4f})")
            
            # Loss convergence check
            if isinstance(epoch_summary['train_loss'], float) and isinstance(epoch_summary['eval_loss'], float):
                loss_diff = abs(epoch_summary['train_loss'] - epoch_summary['eval_loss'])
                print(f"Loss Difference: {loss_diff:.4f}", end="")
                if loss_diff < 0.1:
                    print(" ✅ (Good convergence)")
                elif loss_diff < 0.2:
                    print(" ⚠️ (Acceptable)")
                else:
                    print(" ❌ (Check for overfitting)")
            
            # Target accuracy check
            if isinstance(epoch_summary['eval_accuracy'], float):
                acc = epoch_summary['eval_accuracy']
                if 0.70 <= acc <= 0.80:
                    print(f"🎯 TARGET ACCURACY ACHIEVED! ({acc:.1%})")
                elif acc > 0.80:
                    print(f"⚠️ Accuracy high ({acc:.1%}) - possible overfitting")
                elif acc < 0.70:
                    print(f"📈 Below target ({acc:.1%}) - training continues")
            
            print(f"Checkpoint saved: {args.output_dir}/checkpoint-{state.global_step}")
            print(f"{'='*60}\n")
        
        # Save progress data after each epoch
        self.save_progress()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                **logs
            }
            self.progress_data["training_history"].append(log_entry)
    
    def save_progress(self):
        """Save progress data to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2, default=str)

class WeightedTrainer(Trainer):
    """Custom Trainer with class weighting and regularization"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss with class weighting and label smoothing"""
        # Handle additional arguments that newer versions of transformers might pass
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Apply class weights with label smoothing for regularization
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                label_smoothing=0.1  # Light label smoothing
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # Use label smoothing even without class weights
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train_model(tokenized_dataset, tokenizer, label_encoder, model_name, output_dir, resume_checkpoint=None):
    """Optimized training for 70-80% accuracy with minimal overfitting"""
    print("Starting optimized training...")

    os.makedirs(output_dir, exist_ok=True)

    num_labels = len(label_encoder.classes_)
    
    # Load model with optimal configuration for our target
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=0.2,  # Moderate dropout
        attention_probs_dropout_prob=0.2,
    )

    # Compute balanced class weights
    train_labels = tokenized_dataset["train"]["labels"]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights}")

    # Optimized training arguments with epoch-based saving and progress tracking
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",  # Save after each epoch
        learning_rate=2e-5,  # Lower learning rate for stability
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=16,
        num_train_epochs=12,  # Optimal epochs for target accuracy
        weight_decay=0.05,  # Higher weight decay for regularization
        warmup_steps=100,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Focus on eval_loss to prevent overfitting
        greater_is_better=False,
        logging_steps=25,
        logging_dir=f'{output_dir}/logs',
        save_total_limit=5,  # Keep last 5 checkpoints
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # Disabled for stability
        report_to="none",
        push_to_hub=False,
        # Additional regularization
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Cosine annealing for better convergence
        # Progress tracking
        logging_first_step=True,
        eval_do_concat_batches=False,
    )

    # Create progress tracking callback
    progress_callback = ProgressTrackingCallback(output_dir)

    # Custom trainer with balanced settings and progress tracking
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_enhanced_metrics,
        class_weights=class_weights,
        callbacks=[
            progress_callback,
            EarlyStoppingCallback(
                early_stopping_patience=4,  # Conservative early stopping
                early_stopping_threshold=0.001  # Small threshold
            )
        ],
    )

    # Training with monitoring
    print(f"Training for up to {training_args.num_train_epochs} epochs...")
    print("Target: 70-80% accuracy with train/eval loss convergence")
    
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    
    print(f"\nFINAL RESULTS:")
    print(f"Test Accuracy: {test_metrics.get('eval_accuracy', 0):.4f}")
    print(f"Test F1 (weighted): {test_metrics.get('eval_f1_weighted', 0):.4f}")
    print(f"Test Loss: {test_metrics.get('eval_loss', 0):.4f}")
    
    # Check if we achieved target
    accuracy = test_metrics.get('eval_accuracy', 0)
    if 0.70 <= accuracy <= 0.85:
        print("✅ Target accuracy range achieved!")
    elif accuracy < 0.70:
        print("⚠️ Accuracy below target. Consider:")
        print("  - Increasing epochs to 15-18")
        print("  - Reducing weight_decay to 0.03")
        print("  - Increasing learning_rate to 3e-5")
    else:
        print("⚠️ Accuracy above target (possible overfitting). Consider:")
        print("  - Reducing epochs to 8-10")
        print("  - Increasing weight_decay to 0.08")
        print("  - Adding more dropout")

    return trainer

def save_enhanced_artifacts(trainer, tokenizer, label_encoder, output_dir):
    """Save model and comprehensive artifacts"""
    print(f"Saving enhanced artifacts to {output_dir}")

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label encoder
    with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Create comprehensive label mapping
    label_names = list(label_encoder.classes_)
    
    # Standard label mapping (ID -> Label)
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    
    # Save label mappings
    with open(f"{output_dir}/label_mapping.json", "w", encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    # Save comprehensive mapping info
    mapping_info = {
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": len(label_names),
        "label_names": label_names,
        "model_type": "mood_classification",
        "created_at": datetime.now().isoformat(),
        "target_accuracy": "70-80%",
        "optimization": "balanced_training_eval_loss"
    }
    
    with open(f"{output_dir}/mapping_info.json", "w", encoding='utf-8') as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)

    # Update config.json with proper label mapping
    config_path = f"{output_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["id2label"] = id2label
        config["label2id"] = label2id
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Create a simple inference script
    inference_script = f'''
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json

def load_mood_classifier():
    """Load the trained mood classifier"""
    model_dir = "{output_dir}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    with open(f"{{model_dir}}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    with open(f"{{model_dir}}/mapping_info.json", "r") as f:
        mapping_info = json.load(f)
    
    return model, tokenizer, label_encoder, mapping_info

def predict_mood(text, model, tokenizer, mapping_info):
    """Predict mood for given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions).item()
    
    predicted_mood = mapping_info["id2label"][str(predicted_class)]
    return predicted_mood, confidence

# Example usage:
# model, tokenizer, label_encoder, mapping_info = load_mood_classifier()
# mood, confidence = predict_mood("I am very happy today!", model, tokenizer, mapping_info)
# print(f"Predicted mood: {{mood}} (confidence: {{confidence:.4f}})")
'''
    
    with open(f"{output_dir}/inference_example.py", "w") as f:
        f.write(inference_script)

    print("All enhanced artifacts saved successfully!")

def create_training_summary(output_dir):
    """Create a comprehensive training summary"""
    progress_file = f"{output_dir}/training_progress.json"
    
    if not os.path.exists(progress_file):
        print("No progress file found.")
        return
    
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    # Best metrics
    if progress_data.get("best_metrics"):
        best = progress_data["best_metrics"]
        print(f"🏆 BEST MODEL:")
        print(f"   Epoch: {best.get('best_epoch', 'N/A')}")
        print(f"   Eval Loss: {best.get('best_eval_loss', 'N/A'):.4f}" if isinstance(best.get('best_eval_loss'), (int, float)) else f"   Eval Loss: {best.get('best_eval_loss', 'N/A')}")
        print(f"   Accuracy: {best.get('best_accuracy', 'N/A'):.4f}" if isinstance(best.get('best_accuracy'), (int, float)) else f"   Accuracy: {best.get('best_accuracy', 'N/A')}")
        print(f"   F1 Score: {best.get('best_f1', 'N/A'):.4f}" if isinstance(best.get('best_f1'), (int, float)) else f"   F1 Score: {best.get('best_f1', 'N/A')}")
    
    # Epoch-by-epoch progress
    if progress_data.get("epoch_summaries"):
        print(f"\n📊 EPOCH-BY-EPOCH PROGRESS:")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Eval Loss':<12} {'Accuracy':<10} {'F1 Score':<10} {'Status':<15}")
        print("-" * 80)
        
        for epoch_data in progress_data["epoch_summaries"]:
            epoch = epoch_data.get('epoch', 'N/A')
            train_loss = epoch_data.get('train_loss', 'N/A')
            eval_loss = epoch_data.get('eval_loss', 'N/A')
            accuracy = epoch_data.get('eval_accuracy', 'N/A')
            f1_score = epoch_data.get('eval_f1_weighted', 'N/A')
            
            # Format numbers
            train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else str(train_loss)
            eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss)
            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            f1_score_str = f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else str(f1_score)
            
            # Status
            status = ""
            if isinstance(accuracy, (int, float)):
                if 0.70 <= accuracy <= 0.80:
                    status = "🎯 TARGET"
                elif accuracy > 0.80:
                    status = "⚠️ HIGH"
                else:
                    status = "📈 IMPROVING"
            
            print(f"{epoch:<6} {train_loss_str:<12} {eval_loss_str:<12} {accuracy_str:<10} {f1_score_str:<10} {status:<15}")
    
    # Save detailed summary
    summary_file = f"{output_dir}/training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PUNJABI MOOD CLASSIFICATION - TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model saved in: {output_dir}\n\n")
        
        if progress_data.get("best_metrics"):
            best = progress_data["best_metrics"]
            f.write(f"BEST PERFORMANCE:\n")
            f.write(f"  Best Epoch: {best.get('best_epoch', 'N/A')}\n")
            f.write(f"  Best Eval Loss: {best.get('best_eval_loss', 'N/A')}\n")
            f.write(f"  Best Accuracy: {best.get('best_accuracy', 'N/A')}\n")
            f.write(f"  Best F1 Score: {best.get('best_f1', 'N/A')}\n\n")
        
        f.write("CHECKPOINTS AVAILABLE:\n")
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoint_dirs:
            f.write(f"  {checkpoint}\n")
    
    print(f"\n📄 Detailed summary saved: {summary_file}")
    print("="*80)

def create_evaluation_report(trainer, tokenized_dataset, label_encoder, output_dir):
    """Create comprehensive evaluation report"""
    print("Creating evaluation report...")
    
    # Get predictions on test set
    test_predictions = trainer.predict(tokenized_dataset["test"])
    y_pred = np.argmax(test_predictions.predictions, axis=1)
    y_true = test_predictions.label_ids
    
    # Classification report
    label_names = list(label_encoder.classes_)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    
    # Save classification report
    with open(f"{output_dir}/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix data
    cm_data = {
        "confusion_matrix": cm.tolist(),
        "labels": label_names,
        "accuracy": accuracy_score(y_true, y_pred),
        "target_achieved": 0.70 <= accuracy_score(y_true, y_pred) <= 0.85
    }
    
    with open(f"{output_dir}/confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)
    
    print("Evaluation report saved!")

def main():
    """Optimized main function for 70-80% accuracy target"""
    DATA_FILE = 'punjabi_songs_with_mood.json'
    MODEL_NAME = "xlm-roberta-base"
    OUTPUT_DIR = "./punjabi_sentiment_model_optimized"

    print("OPTIMIZED PUNJABI MOOD CLASSIFICATION TRAINER")
    print("="*60)
    print("PROGRESS TRACKING FEATURES:")
    print("• Checkpoint after every epoch")
    print("• Real-time progress display")
    print("• Training history saved to JSON")
    print("• Best model tracking")
    print("• Loss convergence monitoring")
    print("• Target accuracy alerts")
    print("="*60)
    print("TARGET: 70-80% Accuracy with Balanced Train/Eval Loss")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Data: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Transliteration: {'Available' if TRANSLITERATION_AVAILABLE else 'Not Available'}")
    print("="*60)
    print("OPTIMIZATIONS:")
    print("• Epochs: 12 (with early stopping)")
    print("• Learning Rate: 2e-5 (conservative)")
    print("• Batch Size: 8 (train), 16 (eval)")
    print("• Weight Decay: 0.05 (regularization)")
    print("• Max Length: 256 (efficient)")
    print("• Dropout: 0.2 (moderate)")
    print("• Label Smoothing: 0.1")
    print("• Scheduler: Cosine Annealing")
    print("="*60)

    try:
        # Load and prepare data
        df = load_data(DATA_FILE)
        dataset, label_encoder = prepare_dataset(df, use_augmentation=True)
        tokenized_dataset, tokenizer = tokenize_dataset(dataset, MODEL_NAME)

        # Train model with optimized settings
        trainer = train_model(
            tokenized_dataset,
            tokenizer,
            label_encoder,
            MODEL_NAME,
            OUTPUT_DIR
        )

        # Save artifacts
        save_enhanced_artifacts(trainer, tokenizer, label_encoder, OUTPUT_DIR)
        
        # Create evaluation report
        create_evaluation_report(trainer, tokenized_dataset, label_encoder, OUTPUT_DIR)
        
        # Create training summary
        create_training_summary(OUTPUT_DIR)

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved in: {OUTPUT_DIR}")
        print("="*60)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
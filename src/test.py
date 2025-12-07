import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import safetensors.torch
import os
import pickle
import json

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def gurmukhi_to_roman(text):
    """Convert Gurmukhi text to Roman transliteration"""
    try:
        return transliterate(text, sanscript.GURMUKHI, sanscript.ITRANS)
    except Exception as e:
        print(f"Transliteration error: {e}")
        return text  # fallback to original text


def load_label_mapping(model_dir):
    label_mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
            # Convert keys to int for safety
            label_mapping = {int(k): v for k, v in label_mapping.items()}
            print(f"Loaded label_mapping.json: {label_mapping}")
            return label_mapping
    else:
        print("label_mapping.json not found.")
        return None


def load_model_and_tokenizer():
    """Load the trained model, tokenizer, and extract labels"""
    MODEL_DIR = "./punjabi_sentiment_model/checkpoint-495"

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found!")

    model_path = os.path.join(MODEL_DIR, "model.safetensors")
    config_path = os.path.join(MODEL_DIR, "config.json")

    required_files = [
        ("model.safetensors", model_path),
        ("config.json", config_path)
    ]

    for file_name, file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file '{file_name}' not found in {MODEL_DIR}")

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying to load tokenizer from base model...")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        base_model = config_data.get('_name_or_path', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading model config...")
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_config(config)

    print("Loading model weights...")
    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict)

    print("Loading label mapping from label_mapping.json...")
    label_mapping = load_label_mapping(MODEL_DIR)
    if label_mapping:
        config.id2label = label_mapping
        labels = [label_mapping[i] for i in sorted(label_mapping.keys())]
    else:
        # Fallback to generic labels
        num_labels = getattr(config, 'num_labels', 2)
        labels = [f"LABEL_{i}" for i in range(num_labels)]

    return model, tokenizer, labels, config


def predict_mood(sentence, model, tokenizer, labels, config, device):
    """Predict mood for a given sentence"""
    # Convert Gurmukhi to Roman if needed
    if any(0x0A00 <= ord(char) <= 0x0A7F for char in sentence):
        print("Detected Gurmukhi script, transliterating...")
        romanized_sentence = gurmukhi_to_roman(sentence)
        print(f"Romanized: {romanized_sentence}")
    else:
        romanized_sentence = sentence

    inputs = tokenizer(
        romanized_sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    all_probs = probs.squeeze().cpu().numpy()

    # Use config.id2label mapping to decode predicted class
    predicted_label = config.id2label.get(predicted_class, f"LABEL_{predicted_class}")

    return predicted_label, confidence, romanized_sentence, all_probs


def print_model_info(labels):
    print("\n" + "="*60)
    print("MODEL TRAINING LABELS INFORMATION")
    print("="*60)
    print(f"Number of classes: {len(labels)}")
    print(f"Labels: {labels}")
    print("="*60)


def main():
    try:
        print("Loading model and tokenizer...")
        model, tokenizer, labels, config = load_model_and_tokenizer()

        print_model_info(labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model.to(device)

        input_sentence = "ਤੂੰ ਮੇਰੀ ਜਿੰਦਗੀ ਚ ਆਇਆ ਤਾਂ ਸਭ ਕੁਝ ਚੰਗਾ ਹੋ ਗਿਆ"

        print(f"\nOriginal Sentence: {input_sentence}")

        predicted_label, confidence, romanized_text, all_probs = predict_mood(
            input_sentence, model, tokenizer, labels, config, device
        )

        print(f"\nPREDICTION RESULTS:")
        print(f"Original Sentence: {input_sentence}")
        if romanized_text != input_sentence:
            print(f"Romanized Text: {romanized_text}")
        print(f"Predicted Mood: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")

        print(f"\nAll class probabilities:")
        for label, prob in zip(labels, all_probs):
            print(f"  {label}: {prob:.4f}")

        print("\n" + "="*50)
        print("Interactive Mode - Enter sentences to classify (type 'quit' to exit)")
        print("="*50)

        while True:
            try:
                user_input = input("\nEnter sentence: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue

                mood, conf, rom_text, probs = predict_mood(
                    user_input, model, tokenizer, labels, config, device
                )

                print(f"Predicted Mood: {mood}")
                print(f"Confidence: {conf:.4f}")

                sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                print("Top predictions:")
                for i in range(min(3, len(labels))):
                    idx = sorted_indices[i]
                    print(f"  {labels[idx]}: {probs[idx]:.4f}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing input: {e}")
                continue

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure the checkpoint-495 directory exists")
        print("2. Check that model.safetensors, config.json, and label_mapping.json are present")
        print("3. Verify that the indic-transliteration package is installed")
        print("4. Make sure you have the required dependencies installed")


if __name__ == "__main__":
    main()

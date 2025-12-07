import os
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import safetensors.torch
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ----- Helper -----
def gurmukhi_to_roman(text):
    try:
        return transliterate(text, sanscript.GURMUKHI, sanscript.ITRANS)
    except Exception as e:
        print(f"Transliteration error: {e}")
        return text

# ----- Load Label Mapping -----
def load_label_mapping(base_dir):
    label_mapping_path = os.path.join(base_dir, "label_mapping.json")
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, "r") as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping.items()}
        labels = [id2label[i] for i in sorted(id2label.keys())]
        return labels, id2label
    else:
        print("label_mapping.json not found.")
        return None, None

# ----- Load Model and Tokenizer -----
def load_model_and_tokenizer():
    model_dir = "./punjabi_mood_model1"
    base_dir = "./punjabi_mood_model1"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    print("Loading config...")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_config(config)

    print("Loading weights...")
    weights_path = os.path.join(model_dir, "model.safetensors")
    state_dict = safetensors.torch.load_file(weights_path)
    model.load_state_dict(state_dict)

    print("Loading label mapping...")
    labels, label_mapping = load_label_mapping(base_dir)
    if labels is None:
        labels = [f"LABEL_{i}" for i in range(config.num_labels)]

    return model, tokenizer, labels, label_mapping

# ----- Prediction -----
def predict_mood(sentence, model, tokenizer, labels, device):
    if any(0x0A00 <= ord(c) <= 0x0A7F for c in sentence):
        sentence = gurmukhi_to_roman(sentence)

    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    predicted_label = labels[pred_id] if pred_id < len(labels) else f"LABEL_{pred_id}"
    return predicted_label, confidence, probs.squeeze().cpu().numpy()

# ----- Main -----
def main():
    model, tokenizer, labels, label_mapping = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nInteractive Mode (type 'exit' to quit):")
    while True:
        try:
            text = input("\nEnter Punjabi sentence: ").strip()
            if text.lower() in ["exit", "quit"]:
                break
            label, conf, all_probs = predict_mood(text, model, tokenizer, labels, device)
            print(f"Predicted Mood: {label}")
            print(f"Confidence: {conf:.4f}")
            print("All class probabilities:")
            for lbl, p in zip(labels, all_probs):
                print(f"  {lbl}: {p:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

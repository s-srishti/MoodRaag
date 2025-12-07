import os
import json
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load the JSON dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna(subset=['lyrics', 'mood'])
    df['lyrics'] = df['lyrics'].astype(str)
    df['mood'] = df['mood'].astype(str)
    return df

def save_label_encoder(df, output_dir):
    """Encode moods and save label artifacts"""
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['mood'])

    os.makedirs(output_dir, exist_ok=True)

    # Save label_encoder.pkl
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save label_mapping.json
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    print("[✓] Saved label_encoder.pkl and label_mapping.json")

def main():
    DATA_FILE = 'punjabi_songs_with_mood.json'           # ✅ Your dataset file
    OUTPUT_DIR = './punjabi_mood_model1'             # ✅ Your model directory

    print("Generating label_encoder and label_mapping from data...")
    df = load_data(DATA_FILE)
    save_label_encoder(df, OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()

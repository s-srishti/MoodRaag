import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EmotionClassifier:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label_map = {i: label for i, label in enumerate([
            'angry', 'energetic', 'happy', 'nostalgic', 'romantic', 'sad', 'spiritual'
        ])}
        self.model.eval()
    def predict_emotion(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        return self.label_map[prediction], probs[0][prediction].item()
class SongDatabase:
    def __init__(self, dataset_path):
        with open(dataset_path, "r", encoding='utf-8') as f:
            self.songs = json.load(f)
    def get_songs_by_mood(self, mood):
        return [song for song in self.songs if song.get('mood') and song['mood'].lower() == mood.lower()]
    def get_all_lyrics(self):
        return [song.get('lyrics', '') for song in self.songs]
    def get_song_by_lyrics(self, lyrics):
        for song in self.songs:
            if song.get('lyrics', '') == lyrics:
                return song
        return None
class EnhancedLyricMatcher:
    def __init__(self, song_database):
        self.song_database = song_database
        self.vectorizer = TfidfVectorizer()
        self._fit_vectorizer()
    def _fit_vectorizer(self):
        lyrics_corpus = self.song_database.get_all_lyrics()
        self.tfidf_matrix = self.vectorizer.fit_transform(lyrics_corpus)
    def find_similar_lyrics(self, input_lyrics, mood, top_n=3):
        input_vec = self.vectorizer.transform([input_lyrics])
        cosine_similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        matching_songs = self.song_database.get_songs_by_mood(mood)
        matching_lyrics = [song.get('lyrics', '') for song in matching_songs]
        matching_indices = [i for i, lyrics in enumerate(self.song_database.get_all_lyrics()) if lyrics in matching_lyrics]
        ranked_indices = sorted(matching_indices, key=lambda i: cosine_similarities[i], reverse=True)
        if not ranked_indices:
            emotion_groups = {
                'sad': ['sad', 'nostalgic'],
                'happy': ['happy', 'energetic'],
                'romantic': ['romantic'],
                'angry': ['angry'],
                'spiritual': ['spiritual'],
                'nostalgic': ['nostalgic', 'sad'],
                'energetic': ['energetic', 'happy']
            }
            fallback_labels = emotion_groups.get(mood, [mood])
            fallback_songs = []
            for label in fallback_labels:
                fallback_songs.extend(self.song_database.get_songs_by_mood(label))
            return random.sample(fallback_songs, min(top_n, len(fallback_songs)))
        top_indices = ranked_indices[:top_n]
        return [self.song_database.get_song_by_lyrics(self.song_database.get_all_lyrics()[i]) for i in top_indices]
class ResponseGenerator:
    def __init__(self):
        self.response_templates = {
            'sad': [
                "Tera dil dukhi lagda ae... {} eh gaana sunn ke shayad changa lage.",
                "Thoda emotional mood lagda... {} sun ke halka feel karega.",
                "Eh wali vibe aa rahi ae... {} sahi rahega tenu."
            ],
            'happy': [
                "Vibe badi happy lagdi... {} try kar!",
                "Masti mood ch lagda... {} perfect hovega!",
                "Laggda ae tu hass reha si... {} sun!"
            ],
            'romantic': [
                "Ishq wala love lagda... {} sun!",
                "Dil romantic ho gaya? {} perfect match ae!",
                "Love vibes mil rahi ne... {} enjoy kar!"
            ],
            'nostalgic': [
                "Purani yaadan aa gayian? {} sun ke aur feel aayegi.",
                "Thoda throwback mood lagda... {} sahi choice ae!",
                "Nostalgia wali vibe aa gayi... {} perfect ae!"
            ],
            'angry': [
                "Thoda gussa lagda... {} naal energy mildi ae!",
                "Aggression feel ho rahi ae... {} perfect release ae!",
                "Angry mode ch? {} changi vibe dinda ae!"
            ],
            'spiritual': [
                "Ruhaniyat wali feeling aa rahi... {} sun ke mann shant ho jaega!",
                "Bhakti da mood lagda... {} perfect match hai!",
                "Spiritual zone ch hai tu... {} sun le!",
                "Ishwar di yaad ch... {} sahi lagega!"
            ],
            'energetic': [
                "Energy full on hai! {} da gaana naach layi perfect hai!",
                "Josh wale mood ch hai tu... {} sun ke aur pumped up ho ja!",
                "Energetic vibes mil rahe... {} try kar!",
                "Dance floor bula raha... {} sahi gaana hai!"
            ]
        }
    def generate_response(self, mood, song_name):
        templates = self.response_templates.get(mood, ["Here's a song you might like: {}"])
        response = random.choice(templates).format(song_name)
        return response
class PunjabiLyricsChatbot:
    def __init__(self, model_path, dataset_path):
        self.classifier = EmotionClassifier(model_path)
        self.song_db = SongDatabase(dataset_path)
        self.matcher = EnhancedLyricMatcher(self.song_db)
        self.responder = ResponseGenerator()
    def handle_input(self, user_input):
        mood, confidence = self.classifier.predict_emotion(user_input)
        print(f"\n[Mood Detected: {mood} | Confidence: {confidence:.2f}]")
        recommended_songs = self.matcher.find_similar_lyrics(user_input, mood, top_n=1)
        if recommended_songs:
            song = recommended_songs[0]
            song_title = song.get('title', 'Unknown Title')
            artist = song.get('artist', 'Unknown Artist')
            response = self.responder.generate_response(mood, f"{song_title} by {artist}")
            return response
        else:
            return "Koi gaana nahi mil reha. Thoda hor lyrics dasso."
if __name__ == "__main__":
    MODEL_PATH = "./punjabi_mood_model1"
    DATASET_PATH = "./punjabi_songs_with_mood.json"
    chatbot = PunjabiLyricsChatbot(MODEL_PATH, DATASET_PATH)
    print(">> Punjabi Mood Lyrics Bot\nType 'exit' to quit.\n")
    while True:
        user_input = input("Tuhada mood ya lyrics dasso: ")
        if user_input.lower() == "exit":
            print("Alvida! Fer milange!")
            break
        response = chatbot.handle_input(user_input)
        print(response)

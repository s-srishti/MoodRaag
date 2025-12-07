import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import os

class PunjabiMoodAnalyzer:
    def __init__(self):
        # Initialize feature extraction for similarity
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Mood keyword mappings
        self.punjabi_mood_keywords = {
            'happy': [
                'balle', 'shava', 'khushi', 'masti', 'hansi', 'nachna', 'hasna',
                'sukh', 'lagan', 'mehfil', 'rang', 'jhilmil', 'yaari', 'swag',
                'vibe', 'pind', 'saddi', 'paggan', 'celebration'
            ],
            'sad': [
                'dukh', 'gham', 'udas', 'rona', 'ro', 'akela', 'virah', 'pata',
                'hijr', 'bewafa', 'judai', 'tanha', 'dil tutda', 'palayan',
                'pain', 'tears', 'nafrat'  # ← added here
            ],
            'romantic': [
                'ishq', 'pyaar', 'mohabbat', 'dilbar', 'sajna', 'sajan', 'dil',
                'mehboob', 'yaar', 'chann', 'naina', 'milan', 'prem',
                'rabb', 'sohne', 'ankhiyan', 'gallan'
            ],
            'energetic': [
                'bhangra', 'dhol', 'beat', 'rhythm', 'party', 'nach', 'taal',
                'gidda', 'energy', 'zordar', 'sound', 'music', 'vibe', 'dhamaal',
                'bass', 'paggan', 'jatt', 'swag'
            ],
            'spiritual': [
                'waheguru', 'guru', 'nam', 'simran', 'shabad', 'ardaas', 'prayer',
                'divine', 'blessed', 'bhakti', 'akhar', 'bani', 'kirtan',
                'satnam', 'ek onkar', 'dharam', 'seva', 'naam'
            ],
            'nostalgic': [
                'yaad', 'purane', 'old', 'memories', 'pehla', 'pehli', 'jawani',
                'bachpan', 'gaon', 'ghar', 'school', 'college', 'sathi', 'kisse',
                'lamhe', 'mitti', 'zindagi', 'yaari', 'ankhan'
            ],
            'angry': [  # ← new category added
                'gussa', 'naaraz', 'chikna', 'gali', 'bad words', 'fatte', 'thappad',
                'rukawat', 'lal', 'ghusse', 'bhaida', 'toofan', 'jagda', 'goli',
                'naarazgi', 'dhakka', 'krodh', 'bhasm', 'bhadak', 'kaam', 'nafrat'
            ]
        }

    def analyze_mood(self, lyrics):
        """Analyze mood of song lyrics"""
        try:
            cleaned_lyrics = self.clean_lyrics(lyrics)
            mood = self.analyze_keywords(cleaned_lyrics)
            return {'mood': mood}
        except Exception as e:
            print(f"Error analyzing mood: {e}")
            return {'mood': 'neutral'}

    def clean_lyrics(self, lyrics):
        """Clean and preprocess lyrics"""
        if not lyrics:
            return ""
        cleaned = re.sub(r'\s+', ' ', lyrics)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        return cleaned.strip().lower()

    def analyze_keywords(self, lyrics):
        """Analyze lyrics for mood-specific keywords"""
        mood_scores = {}
        for mood, keywords in self.punjabi_mood_keywords.items():
            score = sum(lyrics.lower().count(keyword) for keyword in keywords)
            mood_scores[mood] = score
        if max(mood_scores.values()) > 0:
            return max(mood_scores, key=mood_scores.get)
        return None

    def process_song_dataset(self, songs_data):
        """Process entire dataset and add mood analysis"""
        processed_songs = []
        for song in songs_data:
            mood_analysis = self.analyze_mood(song.get('lyrics', ''))
            song_with_mood = {**song, **mood_analysis}
            processed_songs.append(song_with_mood)
        return processed_songs


# Usage example
if __name__ == "__main__":
    analyzer = PunjabiMoodAnalyzer()
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build full path to JSON file inside src
json_file_path = os.path.join(script_dir, "punjabi_lyricsmint_songs .json")

with open(json_file_path, 'r', encoding='utf-8') as f:
    songs = json.load(f)

# same for output file
output_file_path = os.path.join(script_dir, "punjabi_songs_with_mood.json")

# Analyze moods
processed_songs = analyzer.process_song_dataset(songs)

# Save to dataset
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(processed_songs, f, ensure_ascii=False, indent=2)

print(f"Mood analysis completed. Output saved to: {output_file_path}")

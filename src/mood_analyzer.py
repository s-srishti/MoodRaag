import json
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

class PunjabiMoodAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis pipeline (multilingual)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        
        # Initialize feature extraction for similarity
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Mood mapping
        self.mood_map = {
            'POSITIVE': 'happy',
            'NEGATIVE': 'sad',
            'NEUTRAL': 'calm'
        }
        
        # Punjabi-specific mood keywords
        self.punjabi_mood_keywords = {
            'happy': ['balle', 'shava', 'khushi', 'masti', 'dance', 'celebration'],
            'sad': ['dukh', 'gham', 'udas', 'rona', 'pain', 'tears'],
            'romantic': ['ishq', 'pyaar', 'mohabbat', 'dilbar', 'sajna', 'love'],
            'energetic': ['bhangra', 'dhol', 'beat', 'rhythm', 'dance', 'party'],
            'spiritual': ['waheguru', 'guru', 'prayer', 'divine', 'blessed'],
            'nostalgic': ['yaad', 'purane', 'old', 'memories', 'past']
        }
    
    def analyze_mood(self, lyrics):
        """Analyze mood of song lyrics"""
        try:
            # Clean lyrics
            cleaned_lyrics = self.clean_lyrics(lyrics)
            
            # Get sentiment from transformer model
            sentiment_result = self.sentiment_analyzer(cleaned_lyrics[:512])  # Limit token length
            primary_sentiment = sentiment_result[0]['label']
            confidence = sentiment_result[0]['score']
            
            # Analyze keywords for more specific mood
            keyword_mood = self.analyze_keywords(cleaned_lyrics)
            
            # Combine results
            final_mood = self.combine_mood_analysis(primary_sentiment, keyword_mood, confidence)
            
            return {
                'primary_mood': final_mood,
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'keyword_mood': keyword_mood
            }
            
        except Exception as e:
            print(f"Error analyzing mood: {e}")
            return {'primary_mood': 'neutral', 'sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    def clean_lyrics(self, lyrics):
        """Clean and preprocess lyrics"""
        if not lyrics:
            return ""
        
        # Remove extra whitespace and special characters
        cleaned = re.sub(r'\s+', ' ', lyrics)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        return cleaned.strip().lower()
    
    def analyze_keywords(self, lyrics):
        """Analyze lyrics for mood-specific keywords"""
        mood_scores = {}
        
        for mood, keywords in self.punjabi_mood_keywords.items():
            score = 0
            for keyword in keywords:
                score += lyrics.lower().count(keyword)
            mood_scores[mood] = score
        
        # Return mood with highest score
        if max(mood_scores.values()) > 0:
            return max(mood_scores, key=mood_scores.get)
        return None
    
    def combine_mood_analysis(self, sentiment, keyword_mood, confidence):
        """Combine sentiment analysis with keyword analysis"""
        if keyword_mood and confidence < 0.7:
            return keyword_mood
        
        if sentiment in self.mood_map:
            return self.mood_map[sentiment]
        
        return keyword_mood if keyword_mood else 'neutral'
    
    def process_song_dataset(self, songs_data):
        """Process entire dataset and add mood analysis"""
        processed_songs = []
        
        for song in songs_data:
            mood_analysis = self.analyze_mood(song.get('lyrics', ''))
            
            song_with_mood = {
                **song,
                'mood': mood_analysis['primary_mood'],
                'sentiment_score': mood_analysis['confidence'],
                'detailed_mood': mood_analysis
            }
            
            processed_songs.append(song_with_mood)
        
        return processed_songs
    
    def get_mood_based_recommendations(self, songs_data, target_mood, limit=5):
        """Get song recommendations based on mood"""
        mood_songs = [song for song in songs_data if song.get('mood') == target_mood]
        
        if not mood_songs:
            # Fallback to similar moods
            similar_moods = self.get_similar_moods(target_mood)
            for similar_mood in similar_moods:
                mood_songs.extend([song for song in songs_data if song.get('mood') == similar_mood])
                if len(mood_songs) >= limit:
                    break
        
        return mood_songs[:limit]
    
    def get_similar_moods(self, mood):
        """Get similar moods for fallback recommendations"""
        mood_similarity = {
            'happy': ['energetic', 'romantic'],
            'sad': ['nostalgic', 'calm'],
            'romantic': ['happy', 'calm'],
            'energetic': ['happy', 'romantic'],
            'spiritual': ['calm', 'nostalgic'],
            'nostalgic': ['sad', 'romantic'],
            'calm': ['spiritual', 'romantic']
        }
        return mood_similarity.get(mood, ['happy'])
    
    def get_lyric_similarity_recommendations(self, songs_data, query_lyrics, limit=5):
        """Get recommendations based on lyrical similarity"""
        if not songs_data:
            return []
        
        # Extract all lyrics
        all_lyrics = [song.get('lyrics', '') for song in songs_data]
        all_lyrics.append(query_lyrics)
        
        # Vectorize lyrics
        tfidf_matrix = self.vectorizer.fit_transform(all_lyrics)
        
        # Calculate similarity with query (last item)
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
        
        # Get top similar songs
        similar_indices = similarities.argsort()[-limit-1:-1][::-1]
        
        return [songs_data[i] for i in similar_indices if similarities[i] > 0.1]

# Usage example
if __name__ == "__main__":
    analyzer = PunjabiMoodAnalyzer()
    
    # Load sample data
    with open('punjabi_folk_songs.json', 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    # Process songs with mood analysis
    processed_songs = analyzer.process_song_dataset(songs)
    
    # Save processed data
    with open('punjabi_songs_with_mood.json', 'w', encoding='utf-8') as f:
        json.dump(processed_songs, f, ensure_ascii=False, indent=2)
    
    print("Mood analysis completed!")
import requests
import json

class PunjabiFolkSongClient:
    """Demo client for the Punjabi Folk Song API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def demo_mood_recommendations(self):
        """Demo mood-based recommendations"""
        print("=== Mood-Based Recommendations Demo ===")
        
        # Get available moods
        response = requests.get(f"{self.base_url}/moods")
        moods = response.json()["moods"]
        print(f"Available moods: {', '.join(moods)}")
        
        # Test each mood
        for mood in moods[:3]:  # Test first 3 moods
            print(f"\n--- Songs for mood: {mood} ---")
            response = requests.post(f"{self.base_url}/recommend/mood", 
                                   json={"mood": mood, "limit": 3})
            
            if response.status_code == 200:
                recommendations = response.json()
                for i, song in enumerate(recommendations["songs"], 1):
                    print(f"{i}. {song['title']} - {song['artist']}")
                    print(f"   Mood: {song.get('mood', 'N/A')}")
            else:
                print(f"Error: {response.status_code}")
    
    def demo_lyrics_analysis(self):
        """Demo lyrics analysis"""
        print("\n=== Lyrics Analysis Demo ===")
        
        test_lyrics = [
            "Jugni kehndi ae jugni kehndi ae, Mera ishq vi tu mera yaar vi tu",
            "Balle balle shava shava, Nach le soniye",
            "Sadi gali wich koi nai aunda, Tere bina jee nai lagda"
        ]
        
        for lyrics in test_lyrics:
            print(f"\nAnalyzing: '{lyrics[:50]}...'")
            response = requests.post(f"{self.base_url}/analyze/lyrics", 
                                   json={"lyrics": lyrics})
            
            if response.status_code == 200:
                analysis = response.json()
                print(f"Detected mood: {analysis['primary_mood']}")
                print(f"Confidence: {analysis['confidence']:.2f}")
            else:
                print(f"Error: {response.status_code}")
    
    def demo_search(self):
        """Demo search functionality"""
        print("\n=== Search Demo ===")
        
        search_terms = ["jugni", "love", "ishq"]
        
        for term in search_terms:
            print(f"\nSearching for: '{term}'")
            response = requests.get(f"{self.base_url}/songs/search?q={term}&limit=3")
            
            if response.status_code == 200:
                results = response.json()
                print(f"Found {results['total_count']} songs")
                for i, song in enumerate(results["results"], 1):
                    print(f"{i}. {song['title']} - {song['artist']}")
            else:
                print(f"Error: {response.status_code}")
    
    def run_full_demo(self):
        """Run complete demo"""
        print("🎵 Punjabi Folk Song Recommender Demo 🎵\n")
        
        try:
            self.demo_mood_recommendations()
            self.demo_lyrics_analysis()
            self.demo_search()
            print("\n✅ Demo completed successfully!")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
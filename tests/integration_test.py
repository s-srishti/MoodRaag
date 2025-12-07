import requests
import time
import json

class APIIntegrationTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def wait_for_api(self, max_attempts=30):
        """Wait for API to be ready"""
        for _ in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("API is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        return False
    
    def test_full_workflow(self):
        """Test complete workflow"""
        print("Testing complete workflow...")
        
        # 1. Check API health
        response = requests.get(f"{self.base_url}/health")
        print(f"Health check: {response.status_code}")
        
        # 2. Get available moods
        response = requests.get(f"{self.base_url}/moods")
        moods = response.json()["moods"]
        print(f"Available moods: {moods}")
        
        # 3. Test mood-based recommendation
        if moods:
            mood = moods[0]
            response = requests.post(f"{self.base_url}/recommend/mood", 
                                   json={"mood": mood, "limit": 3})
            recommendations = response.json()
            print(f"Mood recommendations for '{mood}': {len(recommendations['songs'])} songs")
        
        # 4. Test lyrics analysis
        test_lyrics = "Jugni kehndi ae jugni kehndi ae, Mera ishq vi tu"
        response = requests.post(f"{self.base_url}/analyze/lyrics", 
                               json={"lyrics": test_lyrics})
        analysis = response.json()
        print(f"Lyrics analysis result: {analysis['primary_mood']}")
        
        # 5. Test similarity recommendation
        response = requests.post(f"{self.base_url}/recommend/similarity", 
                               json={"lyrics": test_lyrics, "limit": 3})
        similar_songs = response.json()
        print(f"Similar songs: {len(similar_songs['songs'])} found")
        
        # 6. Test search
        response = requests.get(f"{self.base_url}/songs/search?q=jugni")
        search_results = response.json()
        print(f"Search results: {len(search_results['results'])} songs found")
        
        # 7. Get statistics
        response = requests.get(f"{self.base_url}/stats")
        stats = response.json()
        print(f"Dataset stats: {stats['total_songs']} total songs")
        
        print("Integration test completed successfully!")
    
    def performance_test(self):
        """Test API performance"""
        print("Running performance tests...")
        
        endpoints = [
            "/songs?limit=10",
            "/moods",
            "/stats",
            "/health"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = requests.get(f"{self.base_url}{endpoint}")
            end_time = time.time()
            
            print(f"{endpoint}: {response.status_code} - {(end_time - start_time)*1000:.2f}ms")
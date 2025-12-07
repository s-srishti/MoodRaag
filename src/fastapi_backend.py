from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
from src.mood_analyzer import PunjabiMoodAnalyzer
import uvicorn

app = FastAPI(title="Punjabi Folk Song Recommender", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize mood analyzer
analyzer = PunjabiMoodAnalyzer()

# Data storage
songs_data = []

# Pydantic models
class Song(BaseModel):
    title: str
    artist: str
    lyrics: str
    genre: str = "folk"
    mood: Optional[str] = None
    language: str = "punjabi"
    sentiment_score: Optional[float] = None

class MoodRequest(BaseModel):
    mood: str
    limit: int = 5

class LyricsAnalysisRequest(BaseModel):
    lyrics: str

class SimilarityRequest(BaseModel):
    lyrics: str
    limit: int = 5

class RecommendationResponse(BaseModel):
    songs: List[Dict[str, Any]]
    total_count: int
    mood: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load songs data on startup"""
    global songs_data
    
    # Try to load existing processed data
    processed_file = "punjabi_songs_with_mood.json"
    original_file = "punjabi_folk_songs.json"
    
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            songs_data = json.load(f)
        print(f"Loaded {len(songs_data)} processed songs")
    elif os.path.exists(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            raw_songs = json.load(f)
        
        # Process songs with mood analysis
        songs_data = analyzer.process_song_dataset(raw_songs)
        
        # Save processed data
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(songs_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed and loaded {len(songs_data)} songs")
    else:
        # Create sample data if no files exist
        from src.punjabi_scraper import PunjabiFolkSongScraper
        scraper = PunjabiFolkSongScraper()
        raw_songs = scraper.create_sample_dataset()
        songs_data = analyzer.process_song_dataset(raw_songs)
        
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(songs_data, f, ensure_ascii=False, indent=2)
        
        print(f"Created sample dataset with {len(songs_data)} songs")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Punjabi Folk Song Recommender API",
        "version": "1.0.0",
        "total_songs": len(songs_data)
    }

@app.get("/songs", response_model=List[Dict[str, Any]])
async def get_all_songs(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0)
):
    """Get all songs with pagination"""
    start = offset
    end = start + limit
    return songs_data[start:end]

@app.get("/songs/count")
async def get_songs_count():
    """Get total number of songs"""
    return {"total_songs": len(songs_data)}

@app.get("/moods")
async def get_available_moods():
    """Get all available moods in the dataset"""
    moods = set()
    for song in songs_data:
        if song.get('mood'):
            moods.add(song['mood'])
    
    return {"moods": sorted(list(moods))}

@app.post("/recommend/mood", response_model=RecommendationResponse)
async def recommend_by_mood(request: MoodRequest):
    """Get song recommendations based on mood"""
    try:
        recommendations = analyzer.get_mood_based_recommendations(
            songs_data, request.mood, request.limit
        )
        
        return RecommendationResponse(
            songs=recommendations,
            total_count=len(recommendations),
            mood=request.mood
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/lyrics")
async def analyze_lyrics(request: LyricsAnalysisRequest):
    """Analyze mood of provided lyrics"""
    try:
        mood_analysis = analyzer.analyze_mood(request.lyrics)
        return mood_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/similarity", response_model=RecommendationResponse)
async def recommend_by_similarity(request: SimilarityRequest):
    """Get song recommendations based on lyrical similarity"""
    try:
        recommendations = analyzer.get_lyric_similarity_recommendations(
            songs_data, request.lyrics, request.limit
        )
        
        return RecommendationResponse(
            songs=recommendations,
            total_count=len(recommendations)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/songs/search")
async def search_songs(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Search songs by title, artist, or lyrics"""
    query = q.lower()
    results = []
    
    for song in songs_data:
        if (query in song.get('title', '').lower() or 
            query in song.get('artist', '').lower() or 
            query in song.get('lyrics', '').lower()):
            results.append(song)
            
            if len(results) >= limit:
                break
    
    return {
        "results": results,
        "total_count": len(results),
        "query": q
    }

@app.get("/songs/mood/{mood}")
async def get_songs_by_mood(
    mood: str,
    limit: int = Query(10, ge=1, le=50)
):
    """Get songs filtered by specific mood"""
    filtered_songs = [
        song for song in songs_data 
        if song.get('mood', '').lower() == mood.lower()
    ]
    
    return {
        "songs": filtered_songs[:limit],
        "total_count": len(filtered_songs),
        "mood": mood
    }

@app.get("/songs/artist/{artist}")
async def get_songs_by_artist(
    artist: str,
    limit: int = Query(10, ge=1, le=50)
):
    """Get songs by specific artist"""
    filtered_songs = [
        song for song in songs_data 
        if artist.lower() in song.get('artist', '').lower()
    ]
    
    return {
        "songs": filtered_songs[:limit],
        "total_count": len(filtered_songs),
        "artist": artist
    }

@app.post("/songs/add")
async def add_song(song: Song):
    """Add a new song to the dataset"""
    try:
        # Analyze mood of the new song
        mood_analysis = analyzer.analyze_mood(song.lyrics)
        
        new_song = {
            "title": song.title,
            "artist": song.artist,
            "lyrics": song.lyrics,
            "genre": song.genre,
            "language": song.language,
            "mood": mood_analysis['primary_mood'],
            "sentiment_score": mood_analysis['confidence'],
            "detailed_mood": mood_analysis
        }
        
        songs_data.append(new_song)
        
        # Save updated data
        with open("punjabi_songs_with_mood.json", 'w', encoding='utf-8') as f:
            json.dump(songs_data, f, ensure_ascii=False, indent=2)
        
        return {"message": "Song added successfully", "song": new_song}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get dataset statistics"""
    mood_counts = {}
    artist_counts = {}
    
    for song in songs_data:
        mood = song.get('mood', 'unknown')
        artist = song.get('artist', 'unknown')
        
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
    
    return {
        "total_songs": len(songs_data),
        "mood_distribution": mood_counts,
        "top_artists": dict(sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "songs_loaded": len(songs_data)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
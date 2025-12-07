import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
from src.fastapi_backend import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "total_songs" in data

def test_get_songs():
    """Test getting all songs"""
    response = client.get("/songs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5

def test_mood_recommendation():
    """Test mood-based recommendation"""
    response = client.post("/recommend/mood", json={"mood": "happy", "limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert "songs" in data
    assert "total_count" in data

def test_lyrics_analysis():
    """Test lyrics analysis"""
    response = client.post("/analyze/lyrics", json={
        "lyrics": "Jugni kehndi ae jugni kehndi ae, Mera ishq vi tu"
    })
    assert response.status_code == 200
    data = response.json()
    assert "primary_mood" in data

def test_search_songs():
    """Test song search"""
    response = client.get("/songs/search?q=jugni")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data

def test_health_check():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
import uvicorn
from src.fastapi_backend import app

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
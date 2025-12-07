from setuptools import setup, find_packages

setup(
    name="punjabi-folk-song-recommender",
    version="1.0.0",
    description="A mood-based Punjabi folk song recommender with lyrics analysis",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "transformers>=4.35.2",
        "torch>=2.1.1",
        "scikit-learn>=1.3.2",
        "pandas>=2.1.3",
        "numpy>=1.25.2",
        "beautifulsoup4>=4.12.2",
        "requests>=2.31.0",
        "pydantic>=2.5.0",
    ],
    python_requires=">=3.8",
)
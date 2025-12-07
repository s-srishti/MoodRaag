# MoodRaag
MoodRaag is an end-to-end Punjabi music recommendation system that understands user emotions and recommends songs that truly match their vibe. Instead of relying on genre or popularity, it analyzes the emotional depth of Punjabi song lyrics using modern NLP techniques and a fine-tuned IndicBERTv2 model.
The system works as a conversational chatbot—users can type lyrics or describe their mood naturally (even in code-mixed Punjabi), and the bot responds with a personalized song recommendation.

### Key Features
- 4200+ Punjabi songs scraped from LyricsMint and processed into a structured dataset.
- Automatic mood labeling into 7 classes: happy, sad, nostalgic, angry, energetic, romantic, spiritual.
- Fine-tuned IndicBERTv2 (AI4Bharat) for mood classification on regional-language lyrics.
- TF-IDF–based lyric similarity engine to find songs matching both mood and content.
- Chatbot interface that generates human-like Punjabi responses based on detected emotion.
- Fallback recommendation logic for low-confidence or sparse-mood cases.

### Tech Stack
- Python, PyTorch, Transformers (HuggingFace)
- IndicBERTv2-MLM-only for multilingual sentence classification
- BeautifulSoup4 for scraping
- scikit-learn for TF-IDF, similarity, evaluation
- Pandas / JSON for dataset management

### Dataset & Model
- Scraped: Title, Artist, Lyricist, Lyrics, Metadata, URLs, Language.
- Cleaned, normalized, filtered for quality.
- Songs labelled using a custom mood analyzer.    
- Fine-tuning pipeline includes tokenization, weighted training, class balancing, early stopping, and evaluation.
- Achieved ~59% accuracy F1 on mood classification (see model report on page 21) 

### Chatbot Capabilities
- Detects user mood from raw text/lyrics.
- Recommends the top matching Punjabi song.
- Responds in natural conversational Punjabi using mood-specific templates.
- Handles ambiguous mood cases using similarity and fallback logic.

### Example Interaction
User: Tuhada mood ya lyrics dasso: Sad <br>
[Mood Detected: sad | Confidence: 0.23] <br>
Bot: Thoda emotional mood lagda... Violiin by Arshhh sun ke halka feel karega.

### Project Structure

├── scraper.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;              # Punjabi song web scraper

├── model.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                   # Training & evaluation pipeline

├── lyrics-recommender.py &nbsp;&nbsp;&nbsp;&nbsp;     # Chatbot + mood classifier + lyric matcher

├── punjabi_songs.json &nbsp;&nbsp;&nbsp;&nbsp;        # Raw scraped dataset

├── punjabi_songs_with_mood.json &nbsp;&nbsp;&nbsp;&nbsp;      # Labeled dataset


└── punjabi_mood_model/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Saved fine-tuned model


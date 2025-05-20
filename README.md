# Speech Analysis Dashboard

This project analyzes speech transcripts for sentiment and filler word usage, providing an interactive dashboard for visualization.

## Features

- Sentiment analysis using Hugging Face Transformers
- Filler word detection and ratio calculation
- Interactive dashboard with visualizations
- Detailed metrics and statistics

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Streamlit dashboard implementation
- `analysis.py`: Core analysis functions
- `transcript.txt`: Sample conversation transcript
- `requirements.txt`: Project dependencies

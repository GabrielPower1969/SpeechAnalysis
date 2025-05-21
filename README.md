# Transcript Analysis Dashboard

This project is a technical challenge submission for Bones. It analyzes dialogue transcripts for sentiment and filler-word usage, presenting results in an interactive dashboard.

## Challenge Overview
- **Loads** a transcript file (`transcript.txt`) with 12–16 lines of alternating Speaker A/B dialogue, including at least three filler words (e.g., "um", "like", "you know").
- **Analyzes** each turn for:
  - **Sentiment** (positive, negative, or neutral) using the Hugging Face sentiment-analysis pipeline
  - **Filler-word ratio** (number of filler words ÷ total words) using spaCy and regular expressions
- **Renders** an interactive report (Streamlit) showing per-turn metrics, overall averages, and creative visualizations.

## Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd SpeechAnalysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the application
```bash
streamlit run app.py
```

### 5. View the dashboard
Open the local URL provided by Streamlit (usually http://localhost:8501) in your browser.

## Project Structure
- `app.py` — Main Streamlit application
- `analysis_utils.py` — Core NLP and analysis functions
- `ui_components.py` — UI rendering functions for Streamlit
- `constants.py` — Centralized constants and column names
- `transcript.txt` — Sample transcript file (12–16 lines, alternating speakers, ≥3 filler words)
- `requirements.txt` — Project dependencies
- `README.md` — This documentation

## Metrics & Features
- **Transcript Loading:** Reads and parses `transcript.txt` into dialogue turns.
- **Sentiment Analysis:** Classifies each turn as positive, negative, or neutral using Hugging Face Transformers.
- **Filler Word Ratio:** Calculates the ratio of filler words to total words per turn using spaCy and regex.
- **Interactive Dashboard:**
  - **Transcript & Turn Analysis Tab:**
    - Chat-style transcript display
    - Turn-by-turn analysis table (sentiment, score, filler count, ratio)
    - Sentiment and filler ratio charts
    - CSV download for analysis table
  - **Summary Metrics Tab:**
    - Overall conversation metrics (total turns, words, averages)
    - Per-speaker analysis (turns, words, filler ratio, sentiment counts)
    - Sentiment distribution pie chart
    - Word count distribution and trend charts
- **Robust Error Handling:** Handles edge cases (empty files, missing models, upload errors)

## In one extra hour I would add…
- **Interactive Filtering:** Allow users to click on a speaker or sentiment in a chart to filter the transcript and analysis table, making it easier to explore specific conversation patterns.

## Sample Output
See the `/output_demo/` folder for screenshots of the dashboard and example outputs.

## Notes
- All code is modular, well-documented, and follows best practices for clarity and maintainability.
- The app uses only free, open-source tools and models.
- The transcript format is strictly enforced for fairness and comparability.

---

**Thank you for reviewing my submission!**

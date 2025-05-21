# Transcript Analysis Dashboard

<p float="left">
  <img width="300" alt="Turn Analysis Tab" src="https://github.com/GabrielPower1969/SpeechAnalysis/blob/main/output_demo/1%20turn%20analysis.png?raw=true">
  <img width="300" alt="Summary Analysis Tab" src="https://github.com/GabrielPower1969/SpeechAnalysis/blob/main/output_demo/2%20summary%20analysis.png?raw=true">
  <img width="300" alt="Upload & Download Function" src="https://github.com/GabrielPower1969/SpeechAnalysis/blob/main/output_demo/3%20upload_download_function.png?raw=true">
</p>

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
Looking at this transcript analyzer, we've got a solid foundation. The per-turn metrics are there, and the summary stats with those nice visual breakdowns give a good overview. But to really take it up a notch... from a math/analytical standpoint, beyond simple ratios and averages, we could explore something like sentiment trend analysis – maybe using a moving average on sentiment scores across turns to smooth out noise and highlight significant shifts in conversational tone. Or even basic topic modeling using TF-IDF on the dialogue to automatically tag key subjects per speaker. From a UX perspective, while the current charts and tables are clear, true interactivity would be a game-changer. Imagine clicking on a sentiment segment in a speaker's summary and having the main transcript instantly filter to those specific turns. That direct manipulation makes data exploration so much more intuitive. And development-wise, implementing that kind of cross-filtering would mean a bit more complex state management in Streamlit, perhaps leveraging session state more heavily to link component interactions. Plus, for the topic modeling, integrating another library like Scikit-learn would be necessary, and we'd need to think about how to present those keywords effectively in the UI without cluttering it. Each of these threads – deeper math, richer UX, and the dev effort to enable them – builds on the next, pushing it from a good tool to a really insightful one.

## Sample Output
See the `/output_demo/` folder for screenshots of the dashboard and example outputs.

## Notes
- All code is modular, well-documented, and follows best practices for clarity and maintainability.
- The app uses only free, open-source tools and models.
- The transcript format is strictly enforced for fairness and comparability.

---

**Thank you for reviewing my submission!**

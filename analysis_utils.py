import re
import spacy
from transformers import pipeline
import streamlit as st
from typing import List, Dict, Any
from constants import FILLER_WORDS, SENTIMENT_MODEL_NAME, SPACY_MODEL_NAME, \
    COL_TURN_NUM, COL_SPEAKER, COL_DIALOGUE, COL_SENTIMENT_LABEL, COL_SENTIMENT_SCORE, COL_FILLER_COUNT, COL_TOTAL_WORDS, COL_FILLER_RATIO

@st.cache_resource
def load_sentiment_analyzer():
    """Loads and caches the Hugging Face sentiment analysis pipeline."""
    return pipeline(task="sentiment-analysis", model=SENTIMENT_MODEL_NAME)

@st.cache_resource
def load_spacy_model():
    """Loads and caches the spaCy NLP model."""
    return spacy.load(SPACY_MODEL_NAME)

sentiment_analyzer = load_sentiment_analyzer()
nlp = load_spacy_model()

def map_sentiment_label(raw_label: str) -> str:
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
        "NEG": "NEGATIVE",
        "NEU": "NEUTRAL",
        "POS": "POSITIVE"
    }
    return label_map.get(raw_label.upper(), raw_label)

def parse_transcript(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads and parses the transcript file into a list of dialogue turns.
    Each turn is a dictionary with 'speaker' and 'text'.
    """
    turns = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        current_speaker = None
        current_text_lines = []
        turn_counter = 0
        for line in content.split('\n'):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            match = re.match(r"^(Speaker [A-Za-z0-9]+):(.*)", stripped_line)
            if match:
                if current_speaker and current_text_lines:
                    turn_counter += 1
                    turns.append({
                        'id': turn_counter,
                        'speaker': current_speaker,
                        'text': ' '.join(current_text_lines).strip()
                    })
                current_speaker = match.group(1).strip()
                current_text_lines = [match.group(2).strip()]
            elif current_speaker:
                current_text_lines.append(stripped_line)
        if current_speaker and current_text_lines:
            turn_counter += 1
            turns.append({
                'id': turn_counter,
                'speaker': current_speaker,
                'text': ' '.join(current_text_lines).strip()
            })
    except FileNotFoundError:
        st.error(f"Error: Transcript file '{file_path}' not found.")
        return []
    except Exception as e:
        st.error(f"Error parsing transcript: {e}")
        return []
    return turns

def calculate_sentiment(text: str) -> Dict[str, Any]:
    if not text:
        return {'label': 'NEUTRAL', 'score': 0.0}
    try:
        result = sentiment_analyzer(text)[0]
        return {
            'label': map_sentiment_label(result['label']),
            'score': round(float(result['score']), 4)
        }
    except Exception:
        return {'label': 'ERROR', 'score': 0.0}

def calculate_filler_word_stats(text: str) -> Dict[str, Any]:
    if not text:
        return {'count': 0, 'total_words': 0, 'ratio': 0.0}
    doc = nlp(text)
    total_words = len([token for token in doc if not token.is_punct and not token.is_space])
    filler_count = 0
    text_lower = text.lower()
    for filler in FILLER_WORDS:
        filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
    ratio = (filler_count / total_words) if total_words > 0 else 0.0
    return {
        'count': filler_count,
        'total_words': total_words,
        'ratio': round(ratio, 4)
    }

def analyze_transcript_data(transcript_file_path: str = 'transcript.txt') -> list:
    dialogue_turns = parse_transcript(transcript_file_path)
    analysis_results = []
    if not dialogue_turns:
        return analysis_results
    for turn_data in dialogue_turns:
        text = turn_data['text']
        sentiment_data = calculate_sentiment(text)
        filler_stats = calculate_filler_word_stats(text)
        analysis_results.append({
            COL_TURN_NUM: turn_data['id'],
            COL_SPEAKER: turn_data['speaker'],
            COL_DIALOGUE: text,
            COL_SENTIMENT_LABEL: sentiment_data['label'],
            COL_SENTIMENT_SCORE: sentiment_data['score'],
            COL_FILLER_COUNT: filler_stats['count'],
            COL_TOTAL_WORDS: filler_stats['total_words'],
            COL_FILLER_RATIO: filler_stats['ratio']
        })
    return analysis_results

if __name__ == "__main__":
    print("--- Sentiment Analysis Test Cases ---")
    test_texts = [
        "I love this!",  # positive
        "This is terrible.",  # negative
        "It's okay.",  # neutral
        "",  # empty string
        "Um, well, you know, it's kind of, like, fine.",  # neutral with fillers
        "The food was good, but the service was bad.",  # mixed
        "Absolutely amazing!",  # positive
        "I don't know...",  # ambiguous
        "so, um, like, just, you know",  # only fillers
    ]
    for text in test_texts:
        result = calculate_sentiment(text)
        print(f"Text: '{text}' => Sentiment: {result['label']}, Score: {result['score']}")

    print("\n--- Filler Word Ratio Test Cases ---")
    filler_tests = [
        "Um, I was thinking about trying that new restaurant.",  # 1 filler
        "You know, like, it's basically the best place ever.",  # 3 fillers
        "Well, I mean, it's kind of expensive, but the food is good.",  # 3 fillers
        "No fillers here.",  # 0 fillers
        "",  # empty string
        "um um um um",  # all fillers
        "So, so, so, so, so!",  # repeated filler with punctuation
        "Just a regular sentence with nothing special.",  # 1 filler
        "like, like, like, like, like",  # repeated single filler
    ]
    for text in filler_tests:
        result = calculate_filler_word_stats(text)
        print(f"Text: '{text}' => Filler Count: {result['count']}, Total Words: {result['total_words']}, Ratio: {result['ratio']:.2%}") 
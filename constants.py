# constants.py

FILLER_WORDS = {
    "um", "uh", "like", "you know", "well", "so", "basically", "actually",
    "literally", "just", "kind of", "sort of", "i mean", "right", "okay"
}

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SPACY_MODEL_NAME = "en_core_web_sm"

COL_TURN_NUM = "Turn #"
COL_SPEAKER = "Speaker"
COL_DIALOGUE = "Dialogue Text"
COL_SENTIMENT_LABEL = "Sentiment"
COL_SENTIMENT_SCORE = "Score"
COL_FILLER_COUNT = "Filler Words"
COL_TOTAL_WORDS = "Total Words"
COL_FILLER_RATIO = "Filler Ratio" 
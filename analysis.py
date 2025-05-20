from transformers import pipeline
import spacy
import re

# Initialize the sentiment analysis pipeline with a specific model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="af0f99b"
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define filler words
FILLER_WORDS = {
    "um", "uh", "like", "you know", "well", "so", "basically", "actually",
    "literally", "just", "kind of", "sort of", "i mean", "right", "okay"
}

def load_transcript(file_path):
    """
    Load and parse the transcript file into a list of dialogue turns.
    
    Args:
        file_path (str): Path to the transcript file
        
    Returns:
        list: List of dictionaries containing:
            - speaker: The speaker identifier (e.g., 'Speaker A')
            - text: The text content of the turn
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split into turns and clean up
    turns = []
    current_speaker = None
    current_text = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Speaker'):
            if current_speaker and current_text:
                turns.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text)
                })
            current_speaker = line.split(':')[0].strip()
            current_text = [line.split(':', 1)[1].strip()]
        else:
            current_text.append(line)
    
    # Add the last turn
    if current_speaker and current_text:
        turns.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text)
        })
    
    return turns

def compute_sentiment(text):
    """
    Compute sentiment score for a given text using Hugging Face pipeline.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: A dictionary containing:
            - label: The sentiment label ('POSITIVE', 'NEGATIVE', or 'NEUTRAL')
            - score: The confidence score for the prediction (float between 0 and 1)
    """
    # Get sentiment analysis result
    result = sentiment_analyzer(text)[0]
    
    return {
        'label': result['label'],
        'score': result['score']
    }

def compute_filler_ratio(text):
    """
    Compute the ratio of filler words in the text.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: A dictionary containing:
            - filler_count: Number of filler words found
            - total_words: Total number of words (excluding punctuation)
            - ratio: Ratio of filler words to total words
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count filler words using regex for whole word matches
    filler_count = 0
    for filler in FILLER_WORDS:
        # Use word boundaries to ensure we match whole words only
        filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
    
    # Use spaCy for accurate word counting (excluding punctuation)
    doc = nlp(text)
    total_words = len([token for token in doc if not token.is_punct])
    
    # Calculate ratio (avoid division by zero)
    ratio = filler_count / total_words if total_words > 0 else 0
    
    return {
        'filler_count': filler_count,
        'total_words': total_words,
        'ratio': ratio
    }

# Test the functions
if __name__ == "__main__":
    # Test transcript loading
    print("Testing Transcript Loading:")
    try:
        turns = load_transcript('transcript.txt')
        print(f"\nSuccessfully loaded {len(turns)} turns from transcript.txt")
        for turn in turns:
            print(f"\n{turn['speaker']}: {turn['text']}")
    except FileNotFoundError:
        print("Error: transcript.txt not found")
    
    # Test cases for sentiment analysis
    print("\nTesting Sentiment Analysis:")
    test_texts = [
        "I really enjoyed the movie!",
        "The service was terrible.",
        "The weather is okay today."
    ]
    
    for text in test_texts:
        result = compute_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Confidence: {result['score']:.2f}")
    
    # Test cases for filler word analysis
    print("\nTesting Filler Word Analysis:")
    filler_test_texts = [
        "Um, I was thinking about trying that new restaurant.",
        "You know, like, it's basically the best place ever.",
        "Well, I mean, it's kind of expensive, but the food is good."
    ]
    
    for text in filler_test_texts:
        result = compute_filler_ratio(text)
        print(f"\nText: {text}")
        print(f"Filler words found: {result['filler_count']}")
        print(f"Total words: {result['total_words']}")
        print(f"Filler ratio: {result['ratio']:.2%}")

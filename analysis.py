from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

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

# Test the function
if __name__ == "__main__":
    # Test cases
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

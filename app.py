import streamlit as st
import pandas as pd
import plotly.express as px
from analysis import load_transcript, compute_sentiment, compute_filler_ratio

def main():
    st.title("Speech Analysis Dashboard")
    st.write("Analyzing conversation transcript for sentiment and filler words")

    # Load and analyze transcript
    try:
        turns = load_transcript('transcript.txt')
        
        # Create a DataFrame for easier visualization
        results = []
        for turn in turns:
            sentiment = compute_sentiment(turn['text'])
            filler_stats = compute_filler_ratio(turn['text'])
            
            results.append({
                'Speaker': turn['speaker'],
                'Text': turn['text'],
                'Sentiment': sentiment['label'],
                'Sentiment Score': sentiment['score'],
                'Filler Ratio': filler_stats['ratio'],
                'Filler Count': filler_stats['filler_count'],
                'Total Words': filler_stats['total_words']
            })
        
        df = pd.DataFrame(results)

        # Display the transcript
        st.header("Transcript")
        for turn in turns:
            st.write(f"**{turn['speaker']}**: {turn['text']}")

        # Display metrics
        st.header("Analysis Results")
        
        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment_chart = px.bar(df, x='Speaker', y='Sentiment Score', 
                               color='Sentiment', title='Sentiment Analysis by Turn')
        st.plotly_chart(sentiment_chart)

        # Filler Word Analysis
        st.subheader("Filler Word Analysis")
        filler_chart = px.bar(df, x='Speaker', y='Filler Ratio',
                             title='Filler Word Ratio by Turn')
        st.plotly_chart(filler_chart)

        # Overall Statistics
        st.subheader("Overall Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Sentiment Score", 
                     f"{df['Sentiment Score'].mean():.2f}")
            st.metric("Positive Turns", 
                     len(df[df['Sentiment'] == 'POSITIVE']))
        
        with col2:
            st.metric("Average Filler Ratio", 
                     f"{df['Filler Ratio'].mean():.2%}")
            st.metric("Total Filler Words", 
                     df['Filler Count'].sum())

        # Speaker Comparison
        st.subheader("Speaker Comparison")
        speaker_stats = df.groupby('Speaker').agg({
            'Sentiment Score': 'mean',
            'Filler Ratio': 'mean',
            'Filler Count': 'sum',
            'Total Words': 'sum'
        }).round(3)
        
        st.write("Average metrics by speaker:")
        st.dataframe(speaker_stats)

        # Detailed Results Table
        st.subheader("Detailed Results")
        st.dataframe(df)

    except FileNotFoundError:
        st.error("Error: transcript.txt not found")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
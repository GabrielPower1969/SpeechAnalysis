import streamlit as st
import pandas as pd
import plotly.express as px
from constants import (
    COL_TURN_NUM, COL_SPEAKER, COL_DIALOGUE, COL_SENTIMENT_LABEL,
    COL_SENTIMENT_SCORE, COL_FILLER_COUNT, COL_TOTAL_WORDS, COL_FILLER_RATIO
)

def display_transcript_analysis_tab(df_analysis: pd.DataFrame, raw_turns: list):
    """Renders the 'Transcript Analysis' tab."""
    st.header("🗣️ Conversation Transcript")
    if not raw_turns:
        st.warning("No transcript data to display.")
        return
    for turn in raw_turns:
        if turn['speaker'] == "Speaker A":
            st.markdown(f"<div style='text-align: left; margin-bottom: 10px;'><span style='background-color: #e1f5fe; padding: 5px 10px; border-radius: 10px;'><b>{turn['speaker']}:</b> {turn['text']}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: right; margin-bottom: 10px;'><span style='background-color: #e8f5e9; padding: 5px 10px; border-radius: 10px;'><b>{turn['speaker']}:</b> {turn['text']}</span></div>", unsafe_allow_html=True)
    st.header("Turn-by-Turn Analysis")
    if df_analysis.empty:
        st.warning("No analysis data to display.")
        return
    df_display = df_analysis.copy()
    df_display[COL_SENTIMENT_SCORE] = df_display[COL_SENTIMENT_SCORE].apply(lambda x: f"{x:.1%}")
    df_display[COL_FILLER_RATIO] = df_display[COL_FILLER_RATIO].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_display[[COL_TURN_NUM, COL_SPEAKER, COL_DIALOGUE, COL_SENTIMENT_LABEL, COL_SENTIMENT_SCORE, COL_FILLER_COUNT, COL_TOTAL_WORDS, COL_FILLER_RATIO]], height=400)
    # CSV download button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    csv_turn_by_turn = convert_df_to_csv(df_display)
    st.download_button(
        label="📥 Download Turn-by-Turn Analysis (CSV)",
        data=csv_turn_by_turn,
        file_name='turn_by_turn_analysis.csv',
        mime='text/csv',
    )
    st.subheader("Sentiment Score per Turn")
    if not df_analysis.empty and COL_SENTIMENT_SCORE in df_analysis.columns:
        fig_sentiment_turn = px.bar(df_analysis, x=COL_TURN_NUM, y=COL_SENTIMENT_SCORE, color=COL_SENTIMENT_LABEL,
                                    title="Sentiment Score by Turn",
                                    labels={COL_SENTIMENT_SCORE: "Sentiment Score", COL_TURN_NUM: "Turn Number"},
                                    color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'grey', 'positive': 'green', 'negative': 'red', 'neutral': 'grey'})
        st.plotly_chart(fig_sentiment_turn, use_container_width=True)
    st.subheader("Filler Word Ratio per Turn")
    if not df_analysis.empty and COL_FILLER_RATIO in df_analysis.columns:
        fig_filler_turn = px.line(df_analysis, x=COL_TURN_NUM, y=COL_FILLER_RATIO,
                                 title="Filler Word Ratio by Turn", markers=True,
                                 labels={COL_FILLER_RATIO: "Filler Word Ratio (%)", COL_TURN_NUM: "Turn Number"})
        fig_filler_turn.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_filler_turn, use_container_width=True)

def display_summary_metrics_tab(df_analysis: pd.DataFrame):
    """Renders the 'Summary Metrics' tab."""
    st.header("📊 Overall Conversation Metrics")
    if df_analysis.empty:
        st.warning("No analysis data to display for summary.")
        return
    num_turns = len(df_analysis)
    total_words_corpus = df_analysis[COL_TOTAL_WORDS].sum()
    avg_words_turn = total_words_corpus / num_turns if num_turns > 0 else 0
    overall_filler_words = df_analysis[COL_FILLER_COUNT].sum()
    avg_filler_ratio_corpus = overall_filler_words / total_words_corpus if total_words_corpus > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Turns", num_turns, "🔄")
    col2.metric("Total Words", f"{total_words_corpus:,}", "📄")
    col3.metric("Avg Words/Turn", f"{avg_words_turn:.1f}", "🗣️")
    col4.metric("Avg Filler Ratio", f"{avg_filler_ratio_corpus:.1%}", "💬")
    st.markdown("---")
    st.subheader("Per-Speaker Analysis")
    speaker_summary = df_analysis.groupby(COL_SPEAKER).agg(
        Turns=(COL_TURN_NUM, 'count'),
        Total_Words=(COL_TOTAL_WORDS, 'sum'),
        Total_Filler_Words=(COL_FILLER_COUNT, 'sum'),
        Positive_Turns=(COL_SENTIMENT_LABEL, lambda x: (x.str.lower() == 'positive').sum()),
        Neutral_Turns=(COL_SENTIMENT_LABEL, lambda x: (x.str.lower() == 'neutral').sum()),
        Negative_Turns=(COL_SENTIMENT_LABEL, lambda x: (x.str.lower() == 'negative').sum())
    ).reset_index()
    if not speaker_summary.empty:
        speaker_summary['Avg_Words_Turn'] = (speaker_summary['Total_Words'] / speaker_summary['Turns']).round(1)
        speaker_summary['Filler_Ratio'] = ((speaker_summary['Total_Filler_Words'] / speaker_summary['Total_Words'])).apply(lambda x: f"{x:.1%}" if pd.notnull(x) and speaker_summary['Total_Words'].any() > 0 else "0.0%")
        speaker_display_df = speaker_summary.rename(columns={
            COL_SPEAKER: "Speaker",
            "Total_Words": "Total Words",
            "Avg_Words_Turn": "Avg. Words/Turn",
            "Filler_Ratio": "Avg. Filler Ratio",
            "Positive_Turns": "Positive",
            "Neutral_Turns": "Neutral",
            "Negative_Turns": "Negative"
        })
        st.dataframe(speaker_display_df[['Speaker', 'Turns', 'Total Words', 'Avg. Words/Turn', 'Avg. Filler Ratio', 'Positive', 'Neutral', 'Negative']])
        # Debug print for sentiment label values
        st.write("Sentiment label values:", df_analysis[COL_SENTIMENT_LABEL].unique())
        sentiment_speaker_df = speaker_summary.melt(id_vars=[COL_SPEAKER], value_vars=['Positive_Turns', 'Neutral_Turns', 'Negative_Turns'],
                                                    var_name='Sentiment Type', value_name='Count')
        sentiment_speaker_df['Sentiment Type'] = sentiment_speaker_df['Sentiment Type'].str.replace('_Turns', '')
        fig_sentiment_speaker = px.bar(sentiment_speaker_df, x=COL_SPEAKER, y='Count', color='Sentiment Type',
                                       title="Sentiment Distribution by Speaker", barmode='group',
                                       color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'grey'})
        st.plotly_chart(fig_sentiment_speaker, use_container_width=True)
    st.markdown("---")
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = df_analysis[COL_SENTIMENT_LABEL].value_counts().reset_index()
    sentiment_counts.columns = [COL_SENTIMENT_LABEL, 'Count']
    if not sentiment_counts.empty:
        fig_sentiment_pie = px.pie(sentiment_counts, names=COL_SENTIMENT_LABEL, values='Count',
                                   title="Overall Sentiment Distribution",
                                   color=COL_SENTIMENT_LABEL,
                                   color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'grey'})
        st.plotly_chart(fig_sentiment_pie, use_container_width=True)
        for _, row in sentiment_counts.iterrows():
            st.markdown(f"- **{row[COL_SENTIMENT_LABEL]}**: {row['Count']} turns ({row['Count']/num_turns:.1%})")
    st.markdown("---")
    st.subheader("Word Count Analysis")
    min_words = df_analysis[COL_TOTAL_WORDS].min()
    max_words = df_analysis[COL_TOTAL_WORDS].max()
    median_words = df_analysis[COL_TOTAL_WORDS].median()
    avg_words = df_analysis[COL_TOTAL_WORDS].mean()
    col_wc1, col_wc2, col_wc3, col_wc4 = st.columns(4)
    col_wc1.metric("Min Words/Turn", f"{min_words:.0f}")
    col_wc2.metric("Max Words/Turn", f"{max_words:.0f}")
    col_wc3.metric("Median Words/Turn", f"{median_words:.0f}")
    col_wc4.metric("Avg Words/Turn", f"{avg_words:.1f}")
    if not df_analysis.empty:
        # Robust binning for word count distribution
        if max_words <= 10:
            bins = [0, max_words + 1]
            labels = [f'1-{max_words}']
        elif max_words <= 20:
            bins = [0, 10, max_words + 1]
            labels = ['1-10', f'11-{max_words}']
        elif max_words <= 30:
            bins = [0, 10, 20, max_words + 1]
            labels = ['1-10', '11-20', f'21-{max_words}']
        elif max_words <= 50:
            bins = [0, 10, 20, 30, 50, max_words + 1]
            labels = ['1-10', '11-20', '21-30', '31-50', f'51-{max_words}']
        else:
            bins = [0, 10, 20, 30, 50, 100, max_words + 1]
            labels = ['1-10', '11-20', '21-30', '31-50', '51-100', f'101-{max_words}']
        # Ensure bins are unique
        bins = list(dict.fromkeys(bins))
        df_analysis['Word_Count_Range'] = pd.cut(df_analysis[COL_TOTAL_WORDS], bins=bins, labels=labels, right=True, include_lowest=True)
        word_count_dist = df_analysis['Word_Count_Range'].value_counts().sort_index().reset_index()
        word_count_dist.columns = ['Word Count Range', 'Number of Turns']
        fig_word_dist = px.bar(word_count_dist, x='Word Count Range', y='Number of Turns',
                               title="Distribution of Turns by Word Count",
                               labels={'Number of Turns': "Number of Turns"})
        st.plotly_chart(fig_word_dist, use_container_width=True)
        fig_word_trend = px.bar(df_analysis, x=COL_TURN_NUM, y=COL_TOTAL_WORDS,
                                 title="Word Count per Turn",
                                 labels={COL_TOTAL_WORDS: "Number of Words", COL_TURN_NUM: "Turn Number"})
        st.plotly_chart(fig_word_trend, use_container_width=True) 
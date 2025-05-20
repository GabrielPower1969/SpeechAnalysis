import streamlit as st
st.set_page_config(layout="wide", page_title="Transcript Analysis Dashboard", page_icon="🎙️")

import pandas as pd
from analysis_utils import analyze_transcript_data, parse_transcript
from ui_components import display_transcript_analysis_tab, display_summary_metrics_tab
from constants import COL_DIALOGUE

@st.cache_data
def get_analysis_results(file_path: str) -> pd.DataFrame:
    analysis_list = analyze_transcript_data(file_path)
    if not analysis_list:
        return pd.DataFrame()
    return pd.DataFrame(analysis_list)

@st.cache_data
def get_raw_turns(file_path: str) -> list:
    return parse_transcript(file_path)

def main():
    st.title("🎙️ Transcript Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes dialogue turns for sentiment, filler-word usage,
    and provides insights into conversation patterns.
    Upload your transcript file (UTF-8 encoded `.txt`) or use the default.
    The transcript should follow the format: `Speaker A: Dialogue text...` on new lines.
    """)

    uploaded_file = st.sidebar.file_uploader("Upload Transcript File (transcript.txt)", type=["txt"])
    use_default_file = st.sidebar.checkbox("Use default transcript.txt", True)

    transcript_file_path = 'transcript.txt'
    if uploaded_file is not None and not use_default_file:
        with open("temp_transcript.txt", "wb") as f:
            f.write(uploaded_file.getvalue())
        transcript_file_path = "temp_transcript.txt"
        st.sidebar.success(f"Using uploaded file: {uploaded_file.name}")
    elif use_default_file:
        st.sidebar.info("Using default 'transcript.txt'.")
    else:
        st.warning("Please upload a transcript file or select 'Use default transcript.txt'.")
        return

    try:
        df_results = get_analysis_results(transcript_file_path)
        raw_turns_for_display = get_raw_turns(transcript_file_path)
    except Exception as e:
        st.error(f"A critical error occurred during analysis: {e}")
        return

    if df_results.empty and not raw_turns_for_display:
        st.error("Failed to load or parse the transcript. Please check the file format and content.")
        return
    if df_results.empty and raw_turns_for_display:
         st.warning("Transcript parsed, but no analyzable content found for metrics.")
    elif df_results.empty:
        st.error("Analysis did not produce any results. The transcript might be empty or in an unexpected format.")
        return

    tab_titles = ["💬 Transcript & Turn Analysis", "📊 Summary Metrics"]
    if df_results.empty:
        tab_titles = ["💬 Transcript Display"]

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        if df_results.empty and raw_turns_for_display:
            st.header("🗣️ Conversation Transcript")
            for turn in raw_turns_for_display:
                 st.markdown(f"**{turn['speaker']}**: {turn['text']}")
        elif not df_results.empty:
            display_transcript_analysis_tab(df_results, raw_turns_for_display)
        else:
            st.info("No transcript content to display.")

    if len(tabs) > 1:
        with tabs[1]:
            if not df_results.empty:
                display_summary_metrics_tab(df_results)
            else:
                st.info("No analysis data available for summary metrics.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About this App:**")
    st.sidebar.info("""
    This app performs sentiment and filler-word analysis on conversation transcripts.
    Built with Python, Streamlit, Hugging Face Transformers, and spaCy.
    """)

if __name__ == "__main__":
    main() 
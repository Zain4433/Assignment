import streamlit as st
import whisper
import torch
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load Whisper Model (Make sure torch is installed)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

# Title
st.title("🎙 Whisper AI Transcription & Sentiment Analysis")

# Upload File
uploaded_file = st.file_uploader("Upload an audio file...", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    # Save and transcribe
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Transcribing... ⏳")
    result = model.transcribe("temp_audio.mp3")
    transcribed_text = result["text"]

    # Display Transcribed Text
    st.subheader("📜 Transcription")
    st.write(transcribed_text)

    # Sentiment Analysis
    blob = TextBlob(transcribed_text)
    sentiment_score = blob.sentiment.polarity
    sentiment_category = "Positive" if sentiment_score > 0 else "Neutral" if sentiment_score == 0 else "Negative"

    # Emoji Representation
    emoji = "😃" if sentiment_score > 0 else "😐" if sentiment_score == 0 else "😡"

    # Color-Coded Sentiment
    sentiment_color = "green" if sentiment_score > 0 else "gray" if sentiment_score == 0 else "red"

    # Display Sentiment
    st.subheader("🧠 Sentiment Analysis")
    st.markdown(f"<h3 style='color:{sentiment_color};'>{emoji} {sentiment_category} (Score: {sentiment_score:.2f})</h3>", unsafe_allow_html=True)

    # Generate Word Cloud
    st.subheader("🌟 Word Cloud of Key Phrases")
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(transcribed_text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Summarization (Optional)
    st.subheader("📝 Summary of Transcription")
    summary = " ".join(transcribed_text.split()[:50]) + "..."  # Simple summary by truncating text
    st.write(summary)

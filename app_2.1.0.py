import streamlit as st
import os
import time  # Make sure to import the time module
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
import whisper
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up Streamlit
st.set_page_config(
    page_title="TEASER-MAKER",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set path to FFmpeg (make sure this path is correct)
os.environ["PATH"] += os.pathsep + r"C:\FFmpeg\ffmpeg-2024-08-07-git-94165d1b79-full_build\execes"

# Audio extraction
def extract_audio(video_path, output_path='audio.wav'):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path, codec='pcm_s16le')
    return output_path

# Audio to text
def transcribe_audio(audio_path, model_name='base'):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']

# Summarization
def summarize_text(text, model_name='facebook/bart-large-cnn', max_length=150):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def summarize_long_text(text, model_name='facebook/bart-large-cnn', max_chunk_length=1024):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    tokens = tokenizer.encode(text)
    summaries = []

    for i in range(0, len(tokens), max_chunk_length):
        chunk = tokens[i:i + max_chunk_length]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarize_text(chunk_text, model_name=model_name)
        summaries.append(summary)

    combined_summary = " ".join(summaries)
    return combined_summary

# Load emotion detection model
emotion_classifier = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", return_all_scores=True)

# Emotion detection
def detect_emotions(transcript):
    sentences = transcript.split('. ')
    emotions = []
    
    for sentence in sentences:
        emotion_scores = emotion_classifier(sentence)
        # Convert list of dicts to dict with emotions as keys and scores as values
        emotion_scores = {item['label']: item['score'] for item in emotion_scores[0]}
        emotions.append(emotion_scores)
    
    return emotions

def identify_key_segments_emotion(transcript, emotions, video_duration, num_segments=5, segment_length=10):
    sentences = transcript.split('. ')
    emotion_scores = np.array([max(emotion.values()) for emotion in emotions])

    # Get indices of the top num_segments sentences based on emotion scores
    important_sentences = np.argsort(emotion_scores)[-num_segments:]
    important_sentences.sort()

    segments = []
    time_per_sentence = video_duration / len(sentences)

    last_end_time = 0

    for sentence_index in important_sentences:
        start_time = sentence_index * time_per_sentence
        end_time = min(start_time + segment_length, video_duration)
        
        if start_time >= last_end_time:
            segments.append((start_time, end_time))
            last_end_time = end_time

    return segments

def create_teaser(video_path, segments, output_path='teaser.mp4'):
    video = VideoFileClip(video_path)
    clips = [video.subclip(start, end) for start, end in segments]
    teaser = concatenate_videoclips(clips)
    teaser.write_videofile(output_path, codec='libx264', audio_codec='aac')
    return output_path  # Return the teaser video path

# Function to download YouTube video
def download_youtube_video(url, output_folder='videos'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_folder, 'downloaded_video.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)
    return video_path

# Streamlit app
def main():
    st.title("TEASER-MAKER")

    # Select between uploading file and providing YouTube link
    option = st.selectbox("Choose option", ["Upload Local Video File", "Provide YouTube Video Link"])

    if option == "Upload Local Video File":
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi'])
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            timestamp = int(time.time())
            video_path = f'uploaded_video_{timestamp}{file_extension}'
            audio_path = f'audio_{timestamp}.wav'
            teaser_path = f'teaser_{timestamp}.mp4'

            with open(video_path, 'wb') as f:
                f.write(uploaded_file.read())

            st.video(video_path)

    elif option == "Provide YouTube Video Link":
        youtube_url = st.text_input("Enter YouTube video URL")
        if youtube_url:
            video_path = download_youtube_video(youtube_url)
            file_extension = os.path.splitext(video_path)[1]
            audio_path = f'audio_{int(time.time())}.wav'
            teaser_path = f'teaser_{int(time.time())}.mp4'

            st.video(video_path)

    if st.button("Process Video"):
        if option == "Upload Local Video File" or option == "Provide YouTube Video Link":
            audio_path = extract_audio(video_path, output_path=audio_path)
            transcript = transcribe_audio(audio_path)
            st.write("*Transcript:*")
            st.text(transcript)

            token_count = len(transcript.split())
            st.write(f"*Number of tokens in the transcript:* {token_count}")

            if token_count > 1024:
                st.write("The transcript is too long. Summarizing in smaller chunks.")
                summary = summarize_long_text(transcript)
            else:
                summary = summarize_text(transcript)
            st.write("*Summary:*")
            st.text(summary)

            # Emotion detection
            emotions = detect_emotions(transcript)
            st.write("*Emotion Scores:*")
            st.json(emotions)

            video = VideoFileClip(video_path)
            video_duration = video.duration
            teaser_path = create_teaser(video_path, identify_key_segments_emotion(transcript, emotions, video_duration, num_segments=5), output_path=teaser_path)
            st.write("*Teaser Video:*")
            st.video(teaser_path)

if __name__ == "__main__":
    main()

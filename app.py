import streamlit as st
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from transformers import BartTokenizer, BartForConditionalGeneration
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
#  You need to replace 'C:\FFmpeg...' with the correct path to your FFmpeg executable
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

def identify_key_segments(transcript, video_duration, num_segments=5, segment_length=10):
    sentences = transcript.split('. ')
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    scores = cosine_similarity(vectors)
    scores = np.mean(scores, axis=1)

    # Get indices of the top num_segments important sentences
    important_sentences = np.argsort(scores)[-num_segments:]
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

# Streamlit app
def main():
    st.title("TEASER-MAKER")

    # Contributions section at the top
    st.markdown(
        """
        ## CREDITS:
        CHECK OUT "[SANJAY KUMAR K](https://www.linkedin.com/in/sanjaykumar13055)"
        """
    )

    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi'])

    if uploaded_file is not None:
        # Get the video file extension
        file_extension = os.path.splitext(uploaded_file.name)[1]

        # Generate unique filenames using a timestamp
        import time
        timestamp = int(time.time())
        video_path = f'uploaded_video_{timestamp}{file_extension}'
        audio_path = f'audio_{timestamp}.wav'
        teaser_path = f'teaser_{timestamp}.mp4' # Unique teaser video name

        # Save the video file
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Display the uploaded video
        st.video(video_path)

        # Process the video
        audio_path = extract_audio(video_path, output_path=audio_path)
        transcript = transcribe_audio(audio_path)
        st.write("**Transcript:**")
        st.text(transcript)

        token_count = len(transcript.split())
        st.write(f"**Number of tokens in the transcript:** {token_count}")

        if token_count > 1024:
            st.write("The transcript is too long. Summarizing in smaller chunks.")
            summary = summarize_long_text(transcript)
        else:
            summary = summarize_text(transcript)
        st.write("**Summary:**")
        st.text(summary)

        # Create teaser video
        video = VideoFileClip(video_path)
        video_duration = video.duration
        teaser_path = create_teaser(video_path, identify_key_segments(transcript, video_duration, num_segments=5), output_path=teaser_path)
        st.write("**Teaser Video:**")
        st.video(teaser_path)

if __name__ == "__main__":
    main()

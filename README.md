# TEASER-MAKER🎦 
# TOOLS AND TECHNOLOGIES USED:
1.Streamlit
This python library is used to create an interactive webapplication. It enables users to upload
video files and view the results, including the original video, transcript, summary, and teaser
video. Streamlit facilitates the building of a user-friendly interface for the project.
2. MoviePy:
A powerful Python library for video editing. MoviePy is used to extract audio from video files
and create teaser videos by concatenating key video segments. It handles video and audio
manipulation tasks efficiently.
3. OpenAI's Whisper:
A state-of-the-art speech-to-text model that transcribes audio content into text. Whisper is
employed to convert the audio extracted from the video into a transcript, which is a crucial
step for further processing.
4. Transformers (Hugging Face):
Specifically, the BART (Bidirectional and Auto-Regressive Transformers) model is used
for summarizing the text. This model is utilized to generate concise summaries of the
transcribed text, helping to condense lengthy transcripts into more digestible content.
The Teaser-Maker project is designed to create teaser videos from longer video files by
integrating advanced audio processing and text summarization techniques. The application
extracts audio from uploaded videos using moviepy,transcribes it with OpenAI's Whisper
model, and summarizes the resulting text with the BART model. It identifies key segments of the
video based on the transcript using TF-IDF and cosine similarity, then combines these segments
into a concise teaser video. The user-friendly interface allows for seamless uploading, viewing of
transcripts and summaries, and downloading of the final teaser.
6. Scikit-learn:
A machine learning library that provides tools for vectorizing text using TF-IDF and computing
cosine similarity. Scikit-learn is used to identify key segments of the video based on the
importance of sentences in the transcript.
5. FFmpeg:
A versatile command-line tool for handling multimedia data. Although not directly used in
the code, FFmpeg is essential for audio extraction, and its path is set in the environment
variables to ensure compatibility with MoviePy’s audio processing functions.
# ARCHITECTURE DIAGRAM OF IMPLEMENTATION:
<img src="https://drive.google.com/file/d/1Dx0MDxJqNuB1d59nCiC6RSnkNwhBphFV/view" alt="ARCHITECTURE" width="300">
# DEMO VIDEO
<img src="https://drive.google.com/file/d/1cc9MVgFl184H-obXOzYB-LYic3p_LcJq/view" alt="DEMO" width="300">

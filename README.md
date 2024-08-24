# TEASER-MAKER 🎦

## TOOLS AND TECHNOLOGIES USED:

1. **Streamlit**  
   This Python library is used to create an interactive web application. It enables users to upload video files and view the results, including the original video, transcript, summary, and teaser video. Streamlit facilitates the building of a user-friendly interface for the project.

2. **MoviePy**  
   A powerful Python library for video editing. MoviePy is used to extract audio from video files and create teaser videos by concatenating key video segments. It handles video and audio manipulation tasks efficiently.

3. **OpenAI's Whisper**  
   A state-of-the-art speech-to-text model that transcribes audio content into text. Whisper is employed to convert the audio extracted from the video into a transcript, which is a crucial step for further processing.

4. **Transformers (Hugging Face)**  
   Specifically, the BART (Bidirectional and Auto-Regressive Transformers) model is used for summarizing the text. This model is utilized to generate concise summaries of the transcribed text, helping to condense lengthy transcripts into more digestible content.

5. **Scikit-learn**  
   A machine learning library that provides tools for vectorizing text using TF-IDF and computing cosine similarity. Scikit-learn is used to identify key segments of the video based on the importance of sentences in the transcript.

6. **FFmpeg**  
   A versatile command-line tool for handling multimedia data. Although not directly used in the code, FFmpeg is essential for audio extraction, and its path is set in the environment variables to ensure compatibility with MoviePy’s audio processing functions.
   
7. **Yt-dlp** 
For downloading YouTube videos by giving the link of the Video which will access the youtube and process it

## PROJECT OVERVIEW

The Teaser-Maker project is designed to create teaser videos from longer video files by integrating advanced audio processing and text summarization techniques. The application extracts audio from uploaded videos using MoviePy, transcribes it with OpenAI's Whisper model, and summarizes the resulting text with the BART model. It identifies key segments of the video based on the transcript using TF-IDF and cosine similarity, then combines these segments into a concise teaser video. The user-friendly interface allows for seamless uploading, viewing of transcripts and summaries, and downloading of the final teaser.

## CONTRIBUTIONS
1.[KEERTHANA S](https://github.com/keerthu16)

2.[LEKHA S](https://github.com/lekha0612)

3.[SANJAY KUMAR K](https://github.com/Sanjaykumar13055)

4.[DHANUSH S](https://github.com/Dh4nu5h)

5.[RAHUL A](https://github.com/Rahuleey)

## ARCHITECTURE DIAGRAM OF IMPLEMENTATION

![TEASER-MAKER ARCHITECTURE](https://github.com/Sanjaykumar13055/TEASER-MAKER/blob/7218e1587435c902b94a25c08ac54bcc8a6e2c51/Assests/Teaser-Maker%20Architecture.jpg)

[AUDIO EXTRACTION](https://drive.google.com/file/d/1BHDedW9_NvZ_rSBsiUHvcLfzFajR21Bk/view?usp=sharing)

[AUDIO TRANSCRIPTION](https://drive.google.com/file/d/1OBo3FhA4HeAa-5IIdGD--OiuLAOuejr2/view?usp=sharing)

[KEY SEGMENTS](https://drive.google.com/file/d/1oUmmS-HDzrnuYXncZdzSIpDvokX-baqj/view?usp=sharing)

## KEY FEATURES

1. **Video Processing:** Handles both local files and YouTube links.
2. **Audio Extraction:** Converts video to audio for text transcription.
3. **Transcript Generation:** Utilizes Whisper AI to convert audio to
text.
4. **Text Summarization:** Summarizes lengthy transcripts using the
BART model.
5. **Emotion Detection:** Identifies emotional peaks in the transcript to
determine key moments.
6. **Teaser Creation:** Automatically assembles the most emotionally
impactful and important segments into a teaser video.

## DEMO VIDEO

[Watch Demo Video](https://drive.google.com/file/d/1cc9MVgFl184H-obXOzYB-LYic3p_LcJq/view?usp=sharing)

## OUTPUTS
![TEASER1](https://github.com/Sanjaykumar13055/TEASER-MAKER/blob/7218e1587435c902b94a25c08ac54bcc8a6e2c51/Assests/Teaser%20Output-1.png)

![TEASER5](https://github.com/Sanjaykumar13055/TEASER-MAKER/blob/7218e1587435c902b94a25c08ac54bcc8a6e2c51/Assests/Teaser-Output-6.png)

![TEASER2](https://github.com/Sanjaykumar13055/TEASER-MAKER/blob/7218e1587435c902b94a25c08ac54bcc8a6e2c51/Assests/Teaser-Output-2.png)

![TEASER3](https://github.com/Sanjaykumar13055/TEASER-MAKER/blob/7218e1587435c902b94a25c08ac54bcc8a6e2c51/Assests/Teaser-Output-3.png)

![TEASER4](https://github.com/Sanjaykumar13055/TEASER-MAKER/blob/7218e1587435c902b94a25c08ac54bcc8a6e2c51/Assests/Teaser-Output-5.png)

## OUTPUT VIDEO
[Watch Output Video](https://drive.google.com/file/d/1z_C1UVrEV46cqvhMa5ThbykusCXLIp4V/view)




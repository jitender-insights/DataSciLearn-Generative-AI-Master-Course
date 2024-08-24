import streamlit as st 

#Playing a audio file
st.title("Audio Player")
audio_file = open("media/audio_test.mp3",'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes,format='audio/mp3')


#Playing a Video
st.title("Video Player")
video_file = open("media/video_01_test.mp4",'rb')
video_bytes = video_file.read()

st.video(video_bytes,format='video/mp4')


# Working with Youtube Video

st.title("Youtube Video")
st.video("https://www.youtube.com/watch?v=-VumErLAUGA")


# Uploading and playing an audio file
st.title("Upload and Play Audio")
uploaded_audio = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format='audio/mp3')
    
    
# Uploading and playing a video file
st.title("Upload and Play Video")
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video, format='video/mp4')



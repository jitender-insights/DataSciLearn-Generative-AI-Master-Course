import streamlit as st
import pygame
from pathlib import Path

# Initialize Pygame mixer for audio control
pygame.mixer.init()

# Define paths to audio and video files
AUDIO_FILE_PATH = Path('media/audio_test.mp3')
VIDEO_FILE_PATH = Path('media/video_01_test.mp4')

# Function to handle audio control using Pygame
def handle_audio_control():
    st.header("Audio Player with Controls")

    # Play, Pause, Unpause, and Stop buttons
    if st.button("Play"):
        pygame.mixer.music.load(AUDIO_FILE_PATH)
        pygame.mixer.music.play()

    if st.button("Pause"):
        pygame.mixer.music.pause()

    if st.button("Unpause"):
        pygame.mixer.music.unpause()

    if st.button("Stop"):
        pygame.mixer.music.stop()

    # Volume control
    volume = st.slider("Volume", 0.0, 1.0, 0.5)
    pygame.mixer.music.set_volume(volume)

# Function to display video with custom controls
def handle_video_control():
    st.header("Video Player with Controls")

    # Video display
    video_file = open(VIDEO_FILE_PATH, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4')

    # Play/Pause control (not supported directly by Streamlit, but shown for interface design)
    if st.button("Play Video"):
        st.write("Playing video... (use the video player's controls)")
    if st.button("Pause Video"):
        st.write("Pausing video... (use the video player's controls)")

# Main function to create the Streamlit app
def main():
    st.title("Interactive Audio and Video Player")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Home", "Audio Player", "Video Player"])

    if choice == "Home":
        st.subheader("Welcome to the Interactive Audio and Video Player App")
        st.write("""
        This app allows you to play and control audio and video files using 
        custom widgets. Navigate to the respective sections using the sidebar.
        """)
    elif choice == "Audio Player":
        handle_audio_control()
    elif choice == "Video Player":
        handle_video_control()

if __name__ == "__main__":
    main()

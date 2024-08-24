import streamlit as st 
import pygame


# Initialize Pygame mixer
pygame.mixer.init()

#Load an audio file
audio_file = "media/audio_test.mp3"
pygame.mixer.music.load(audio_file)

#Streamlit interface for control buttons
st.title("Advanced Audio Controls")

if st.button("Play"):
    pygame.mixer.music.play()
    
if st.button("Pause"):
    pygame.mixer.music.pause()
    
if st.button("Unpause"):
    pygame.mixer.music.unpause()
    
if st.button("Stop"):
    pygame.mixer.music.stop()
    
    
#Volume slider
volume = st.slider("Volume",0.0,1.0,0.5) 
pygame.mixer.music.set_volume(volume)  
    

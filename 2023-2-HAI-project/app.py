import streamlit as st
import numpy as np
from MusicGen import MusicGen

sample_rate = 32000  # {sample_rate} samples per second
music_gen = MusicGen(sample_rate)

st.header('Music Generator')

prompt = st.text_area('Prompt',
                placeholder='Write text to describe the music you want.'
            )

st.button("Reset", type="primary")

if st.button('Submit'):
    music_gen.generate_music(prompt)
    audio_file = open('sample_audio/test0.mp3', 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/mp3')

    for i in range(1, 11):
        if st.button("stop", key=i, type="primary") is False:
            music_gen.regenerate_music(i)
            path = f'sample_audio/test{i}.mp3'
            audio_file = open(path, 'rb')
            audio_bytes = audio_file.read()
            
            st.audio(audio_bytes, format='audio/mp3')

else:
    st.write('Music Player Here')
import streamlit as st
import numpy as np
from MusicGen import MusicGen

sample_rate = 24000  # {sample_rate} samples per second
music_gen = MusicGen(sample_rate)

st.header('Music Generator')

prompt1 = st.text_area('1st Prompt',
                      placeholder='Write text to describe the music you want.'
					)

prompt2 = st.text_area('2nd Prompt',
                      placeholder='Write text to describe the music you want.'
					)

st.button("Reset", type="primary")
if st.button('Submit'):
	music_gen.generate_music(prompt1, prompt2)
	audio_file = open('sample_audio/test.mp3', 'rb')
	audio_bytes = audio_file.read()

	st.audio(audio_bytes, format='audio/mp3')

	while st.button("stop", type="primary") is False:
		music_gen.regenerate_music()
		audio_file = open('sample_audio/test.mp3', 'rb')
		audio_bytes = audio_file.read()
else:
    st.write('Music Player Here')
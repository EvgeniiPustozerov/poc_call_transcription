import glob
import random
from subprocess import STDOUT, check_call
import os
from sys import platform

import soundfile as sf
import streamlit as st
from pydub import AudioSegment

from modules.diarization.nemo_diarization import diarization


# Prerequisites for the streamlit server

if platform == "linux" or platform == "linux2":
    check_call(['apt-get', 'install', '-y', 'libsndfile1'], stdout=open(os.devnull, 'wb'), stderr=STDOUT)

st.title('Call Transcription demo')
st.subheader('This simple demo shows the possibilities of the ASR and NLP in the task of '
             'automatic speech recognition and diarization. It works with mp3, ogg and wav files. You can randomly '
             'pickup a set of images from the built-in database or try uploading your own files.')

if st.button('Try random samples from the database'):
    folder = "data/datasets/crema_d_diarization_chunks"
    os.makedirs(folder, exist_ok=True)
    list_all_audio = glob.glob("data/datasets/crema_d_diarization_chunks/*.wav")
    chosen_files = sorted(random.sample(list_all_audio, 1))
    file_name = os.path.basename(chosen_files[0]).split(".")[0]
    audio_file = open(chosen_files[0], 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    f = sf.SoundFile(chosen_files[0])
    st.write("Starting transcription. Estimated processing time: %0.1f seconds" % (f.frames / (f.samplerate * 5)))
    result = diarization(chosen_files[0])
    with open("info/transcripts/pred_rttms/" + file_name + ".txt") as f:
        transcript = f.read()
    st.write("Transcription completed.")
    st.write("Number of speakers: %s" % result[file_name]["speaker_count"])
    st.write("Sentences: %s" % len(result[file_name]["sentences"]))
    st.write("Words: %s" % len(result[file_name]["words"]))
    st.download_button(
        label="Download audio transcript",
        data=transcript,
        file_name='transcript.txt',
        mime='text/csv',
    )

uploaded_file = st.file_uploader("Choose your recording with a speech",
                                 accept_multiple_files=False, type=["mp3", "wav", "ogg"])
if uploaded_file is not None:
    folder = "data/user_data/"
    os.makedirs(folder, exist_ok=True)
    for f in glob.glob(folder + '*'):
        os.remove(f)
    save_path = folder + uploaded_file.name
    sound = AudioSegment.from_mp3(uploaded_file)
    sound.export(save_path, format="wav", parameters=["-ac", "1"])
    file_name = os.path.basename(save_path).split(".")[0]
    audio_file = open(save_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    f = sf.SoundFile(save_path)
    st.write("Starting transcription. Estimated processing time: %0.0f minutes and %02.0f seconds"
             % ((f.frames / (f.samplerate * 3) // 60), (f.frames / (f.samplerate * 3) % 60)))
    result = diarization(save_path)
    with open("info/transcripts/pred_rttms/" + file_name + ".txt") as f:
        transcript = f.read()
    st.write("Transcription completed.")
    st.write("Number of speakers: %s" % result[file_name]["speaker_count"])
    st.write("Sentences: %s" % len(result[file_name]["sentences"]))
    st.write("Words: %s" % len(result[file_name]["words"]))
    st.download_button(
        label="Download audio transcript",
        data=transcript,
        file_name='transcript.txt',
        mime='text/csv',
    )

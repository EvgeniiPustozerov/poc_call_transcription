import glob
import random
import os
import numpy as np
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
from datasets import load_dataset
from scipy.io.wavfile import write

from modules.diarization.nemo_diarization import diarization

FOLDER_WAV_DB = "data/database/"
FOLDER_WAV_USER = "data/user_data/"
SAMPLE_RATE = 16000
dataset = load_dataset("pustozerov/crema_d_diarization", split='validation')

st.title('Call Transcription demo')
st.subheader('This simple demo shows the possibilities of the ASR and NLP in the task of '
             'automatic speech recognition and diarization. It works with mp3, ogg and wav files. You can randomly '
             'pickup a set of images from the built-in database or try uploading your own files.')
if st.button('Try a random sample from the database'):
    os.makedirs(FOLDER_WAV_DB, exist_ok=True)
    shuffled_dataset = dataset.shuffle(seed=random.randint(0, 100))
    file_name = str(shuffled_dataset["file"][0]).split(".")[0]
    audio_bytes = np.array(shuffled_dataset["data"][0])
    audio_bytes_scaled = np.int16(audio_bytes / np.max(np.abs(audio_bytes)) * 32767)
    write(os.path.join(FOLDER_WAV_DB, file_name + '.wav'), rate=SAMPLE_RATE, data=audio_bytes_scaled)
    f = sf.SoundFile(os.path.join(FOLDER_WAV_DB, file_name + '.wav'))
    audio_file = open(os.path.join(FOLDER_WAV_DB, file_name + '.wav'), 'rb')
    st.audio(audio_file.read())
    st.write("Starting transcription. Estimated processing time: %0.1f seconds" % (f.frames / (f.samplerate * 5)))
    result = diarization(os.path.join(FOLDER_WAV_DB, file_name + '.wav'))
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
    os.makedirs(FOLDER_WAV_USER, exist_ok=True)
    for f in glob.glob(FOLDER_WAV_USER + '*'):
        os.remove(f)
    save_path = FOLDER_WAV_USER + uploaded_file.name
    if ".mp3" in uploaded_file:
        sound = AudioSegment.from_mp3(uploaded_file)
    elif ".ogg" in uploaded_file:
        sound = AudioSegment.from_ogg(uploaded_file)
    else:
        sound = AudioSegment.from_wav(uploaded_file)
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

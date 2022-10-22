import shutil

import gradio as gr
import random
import os
import numpy as np
from pydub import AudioSegment
from datasets import load_dataset
from scipy.io.wavfile import write

from modules.diarization.nemo_diarization import diarization
from modules.nlp.nemo_ner import detect_ner
from modules.nlp.nemo_punct_cap import punctuation_capitalization

FOLDER_WAV_DB = "data/database/"
FOLDER_USER_DATA = "data/user_data/"
FOLDER_USER_DATA_WAV = "data/user_data_wav/"
FOLDER_MANIFESTS = "info/configs/manifests/"
SAMPLE_RATE = 16000
dataset = load_dataset("pustozerov/crema_d_diarization", split='validation')
os.makedirs(FOLDER_WAV_DB, exist_ok=True)
os.makedirs(FOLDER_MANIFESTS, exist_ok=True)


def process_audio(uploaded_file=None):
    if uploaded_file:
        secondary_audio = False
        folder_wav = FOLDER_USER_DATA_WAV
        os.makedirs(folder_wav, exist_ok=True)
        print(uploaded_file)
        shutil.move(uploaded_file, os.path.join(FOLDER_USER_DATA, os.path.basename(uploaded_file)))
        uploaded_file = os.path.join(FOLDER_USER_DATA, os.path.basename(uploaded_file))
        print(uploaded_file)
        if ".mp3" in uploaded_file:
            sound = AudioSegment.from_mp3(uploaded_file)
        elif ".ogg" in uploaded_file:
            sound = AudioSegment.from_ogg(uploaded_file)
        else:
            sound = AudioSegment.from_wav(uploaded_file)
        save_path = folder_wav + os.path.basename(uploaded_file)
        os.makedirs(folder_wav, exist_ok=True)
        sound.export(save_path, format="wav", parameters=["-ac", "1"])
        file_name = os.path.basename(save_path).split(".")[0]
        result = diarization(save_path)
    else:
        secondary_audio = True
        folder_wav = FOLDER_WAV_DB
        os.makedirs(folder_wav, exist_ok=True)
        shuffled_dataset = dataset.shuffle(seed=random.randint(0, 100))
        file_name = str(shuffled_dataset["file"][0]).split(".")[0]
        audio_bytes = np.array(shuffled_dataset["data"][0])
        audio_bytes_scaled = np.int16(audio_bytes / np.max(np.abs(audio_bytes)) * 32767)
        write(os.path.join(folder_wav, file_name + '.wav'), rate=SAMPLE_RATE, data=audio_bytes_scaled)
        result = diarization(os.path.join(folder_wav, file_name + '.wav'))
    transcript_path = "info/transcripts/pred_rttms/" + file_name + ".txt"
    with open(transcript_path) as f:
        transcript = f.read()
    sentences = result[file_name]["sentences"]
    all_strings = ""
    for sentence in sentences:
        all_strings = all_strings + sentence["sentence"] + "\n"
    all_strings = punctuation_capitalization([all_strings])[0]
    tagged_string, tags_summary = detect_ner(all_strings)
    transcript = transcript + '\n' + tagged_string
    with open(transcript_path, 'w') as f:
        f.write(transcript)
    output = "<p>Number of speakers: %s" % result[file_name]["speaker_count"] + "<br>" \
             + "Sentences: %s" % len(result[file_name]["sentences"]) + "<br>" \
             + "Words: %s" % len(result[file_name]["words"]) + "<br>" \
             + "Found named entities: %s" % tags_summary + "</p>"
    return [audio_output.update(os.path.join(folder_wav, file_name + '.wav'), visible=secondary_audio),
            output, file_output.update(transcript_path, visible=True)]


with gr.Blocks() as demo:
    gr.HTML('<br><h1><font size="+4">Call Transcription demo</font></h1>')
    gr.HTML('<p><font size="+1">This simple demo shows the possibilities of ASR and NLP in the task of automatic '
            'speech recognition '
            'and diarization. It works with mp3, ogg, and wav files. You can randomly pick an audio file with the '
            'dialogue from the built-in database or try uploading your files.</font></p>')
    gr.Markdown('<p><font size="+1">Note: this demo shows up a reduced-performance model. To get a full-performance '
                'neural network or '
                'develop a system adapted to your task – contact <a '
                'href="mailto:kirill.lozovoi@exposit.com?subject=Request for '
                'information">kirill.lozovoi@exposit.com</a>.</font></p>')
    audio_input = gr.Audio(source="upload", type="filepath")
    second_btn = gr.Button('Try uploaded audiofile')
    gr.Markdown('<center><p>or</p></center>')
    first_btn = gr.Button('Try a random sample from the database')

    # Output zone
    audio_output = gr.Audio(visible=False, interactive=True)
    text_output = gr.HTML()
    file_output = gr.File(label="Download audio transcript", visible=False)

    # noinspection PyTypeChecker
    first_btn.click(fn=process_audio, inputs=None,
                    outputs=[audio_output, text_output, file_output])
    # noinspection PyTypeChecker
    second_btn.click(fn=process_audio, inputs=audio_input, outputs=[audio_output, text_output, file_output])

demo.launch(share=True)

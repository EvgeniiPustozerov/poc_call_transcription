import os

from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
from omegaconf import OmegaConf
from pyannote.audio import Pipeline

ROOT = os.getcwd()
MODEL_CONFIG = "info/configs/offline_diarization_asr.yaml"
data_dir = os.path.join(ROOT, 'info/configs/')
os.makedirs(data_dir, exist_ok=True)
output_dir = os.path.join(ROOT, 'info/transcripts/')
os.makedirs(output_dir, exist_ok=True)


def diarization(file_path):
    # Create a manifest for input with below format.
    # {'audio_filepath': /path/to/audio_file, 'offset': 0, 'duration':None, 'label': 'infer', 'text': '-',
    # 'num_speakers': None, 'rttm_filepath': /path/to/rttm/file, 'uem_filepath'='/path/to/uem/filepath'}
    import json
    meta = {
        'audio_filepath': file_path,
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': 2,
        'rttm_filepath': None,
        'uem_filepath': None
    }
    with open(os.path.join(data_dir, 'manifests/', 'input_manifest.json'), 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')

    # Make a manifest with an external VAD
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    output = pipeline(file_path)
    initial_json = output.for_json()
    keys = ("audio_filepath", "offset", "duration", "label")
    output_json = []
    for segment in initial_json["content"]:
        vad_json = dict.fromkeys(keys)
        vad_json["audio_filepath"] = file_path
        vad_json["offset"] = segment["segment"]["start"]
        vad_json["duration"] = segment["segment"]["end"] - segment["segment"]["start"]
        vad_json["label"] = "SPEECH"
        vad_json["uniq_id"] = initial_json["uri"]
        output_json.append(vad_json)
    with open(os.path.join(data_dir, 'manifests/', 'external_vad_manifest.json'), 'w') as f:
        for item in output_json:
            f.write(str(item).replace("'", '"') + '\n')

    config2 = OmegaConf.load(MODEL_CONFIG)
    config2.diarizer.asr.model_path = 'QuartzNet15x5Base-En'
    config2.diarizer.manifest_filepath = \
        os.path.join(data_dir, 'manifests/', 'input_manifest.json')
    config2.diarizer.speaker_embeddings.model_path = 'titanet_large'
    config2.diarizer.vad.external_vad_manifest = \
        os.path.join(data_dir, 'manifests/', 'external_vad_manifest.json')
    config2.diarizer.out_dir = output_dir
    config2.num_workers = 0
    asr_ts_decoder = ASR_TIMESTAMPS(**config2.diarizer)
    asr_model = asr_ts_decoder.set_asr_model()
    word_hyp, word_ts_hyp = asr_ts_decoder.run_ASR(asr_model)
    print(word_hyp)
    print(word_ts_hyp)

    asr_diar_offline = ASR_DIAR_OFFLINE(**config2.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset
    diar_hyp, diar_score = asr_diar_offline.run_diarization(config2, word_ts_hyp)
    print("Diarization hypothesis output: \n", diar_hyp)
    result = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
    file_to_show = os.path.join(data_dir, 'transcripts/pred_rttms/', file_path.split('/')[-1].split(".")[0], '.txt')
    print(file_to_show)
    print(diar_hyp)
    return result

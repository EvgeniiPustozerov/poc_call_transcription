import os

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
from omegaconf import OmegaConf
from pyannote.audio import Pipeline
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE

ROOT = os.getcwd()
MODEL_CONFIG = "info/configs/diarization_nemo/offline_diarization.yaml"
MODEL_CONFIG_2 = "info/configs/diarization_nemo/offline_diarization_asr.yaml"
data_dir = os.path.join(ROOT, 'info/configs/diarization_nemo/')
os.makedirs(data_dir, exist_ok=True)
output_dir = os.path.join(ROOT, 'info/configs/diarization_nemo/vad/')
output_dir_2 = os.path.join(ROOT, 'info/configs/diarization_nemo/vad_2/')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_2, exist_ok=True)


def rttm_to_diar_hyp():
    pass


def diarization(file_path, with_asr=False):
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
    with open(os.path.join(ROOT, 'info/configs/diarization_nemo/manifests/', 'input_manifest.json'), 'w') as fp:
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
    with open(os.path.join(ROOT, 'info/configs/diarization_nemo/manifests/', 'external_vad_manifest.json'), 'w') as f:
        for item in output_json:
            f.write(str(item).replace("'", '"') + '\n')

    if with_asr:
        config2 = OmegaConf.load(MODEL_CONFIG_2)
        config2.diarizer.asr.model_path = 'QuartzNet15x5Base-En'
        config2.diarizer.manifest_filepath = \
            os.path.join(ROOT, 'info/configs/diarization_nemo/manifests/', 'input_manifest.json')
        config2.diarizer.speaker_embeddings.model_path = 'titanet_large'
        config2.diarizer.vad.external_vad_manifest = \
            os.path.join(ROOT, 'info/configs/diarization_nemo/manifests/', 'external_vad_manifest.json')
        config2.diarizer.out_dir = output_dir_2
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
        print(result)
    else:
        config = OmegaConf.load(MODEL_CONFIG)
        config.diarizer.vad.external_vad_manifest = \
            os.path.join(ROOT, 'info/configs/diarization_nemo/manifests/', 'external_vad_manifest.json')
        pretrained_speaker_model = 'titanet_large'
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.manifest_filepath = \
            os.path.join(ROOT, 'info/configs/diarization_nemo/manifests/', 'input_manifest.json')
        config.diarizer.out_dir = output_dir
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = True
        config.num_workers = 0
        sd_model = ClusteringDiarizer(cfg=config)
        result = sd_model.diarize()
    return result

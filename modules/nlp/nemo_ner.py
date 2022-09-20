import re
from collections import Counter

from nemo.collections import nlp as nemo_nlp

new_config = nemo_nlp.models.TokenClassificationModel.from_pretrained(model_name="ner_en_bert", return_config=True)
new_config.dataset.num_workers = 0
pretrained_ner_model = nemo_nlp.models.TokenClassificationModel.from_pretrained(
    model_name="ner_en_bert", override_config_path=new_config)


def detect_ner(input_string):
    results = pretrained_ner_model.add_predictions([input_string.replace('[', '').replace(']', '')])[0]
    print(results)
    tags = re.findall('\[.*?]', results)
    output_string = "Found named entities: " + str(dict(Counter(tags)))[1:-1]
    return output_string

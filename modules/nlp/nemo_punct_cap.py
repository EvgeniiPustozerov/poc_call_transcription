from nemo.collections.nlp.models import PunctuationCapitalizationModel


punctuation_capitalization_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")


def punctuation_capitalization(text):
    return punctuation_capitalization_model.add_punctuation_capitalization(text)

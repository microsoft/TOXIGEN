from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import random

class HateSpeechClassifier():
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()

    def __call__(self, input_ids, labels=None):
        outputs = self.model(input_ids)
        return outputs

class HateBERT(HateSpeechClassifier):
    def __init__(self, model_path, fname=None):
        """
        HateBERT files: https://huggingface.co/GroNLP/hateBERT
        """
        super(HateBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')
        self.model = AutoModelForSequenceClassification.from_pretrained('GroNLP/hateBERT').eval()

class ToxDectRoBERTa(HateSpeechClassifier):
    def __init__(self):
        """
        ToxDectRoBERTa files: https://huggingface.co/Xuhui/ToxDect-roberta-large
        """
        super(ToxDectRoBERTa, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
        self.model = AutoModelForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large').eval()

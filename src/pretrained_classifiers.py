import os
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
import random

class HateSpeechClassifier(torch.nn.Module):
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()

    def __call__(self, input_ids, labels=None):
        outputs = self.model(input_ids)
        logits = outputs.logits
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            outputs = (loss, logits)
        else:
            outputs = logits.detach()
        return outputs

class HateBERT(torch.nn.Module):
    """
    This is the model we trained on gpt3-generated data.
    Example
    -------
    >>> model = HateBERT(path_to_folder_containing_weights)
    >>> hate_probabilities = model(["sentence 1", "sentence 2"])
    """
    def __init__(self, model_path, fname=None):
        super(HateBERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "pytorch_model.bin"), config=os.path.join(model_path, "config.json"))
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", config=os.path.join(model_path, "tokenizer_config.json"))

class ToxDectRoberta(torch.nn.Module):
    def __init__(self):
        super(ToxDectRoberta, self).__init__()
        config = RobertaConfig.from_pretrained('Xuhui/ToxDect-roberta-large')
        self.tokenizer = RobertaTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
        self.model = RobertaForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large', config=config)

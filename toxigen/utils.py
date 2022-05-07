from toxigen.language_models import GPT3, ALICE
import pandas as pd

def alice(prompt, classifier, endpoint_url, api_key, mode="neutral", num_to_generate=1, device="cpu"):
    language_model = ALICE(GPT3(endpoint_url=endpoint_url, apikey=api_key), classifier, mode, device="cpu")
    
    return [language_model(prompt) for _ in range(num_to_generate)]

def label_annotations(annotated):
    # Annotations should be the annotated dataset
    label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
    labeled_annotations = pd.DataFrame()
    labeled_annotations["text"] = [i for i in annotated.text.tolist()]
    labeled_annotations["label"] = label
    return labeled_annotations
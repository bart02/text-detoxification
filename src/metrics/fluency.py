from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

cola_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
cola_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")


def classify_cola(preds, batch_size=16):
    results = []

    for i in tqdm(range(0, len(preds), batch_size)):
        batch = cola_tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)
        result = (cola_model(**batch)['logits']).softmax(dim=1)[:,1].data.tolist()
        results.extend(result)

    return np.array(results)

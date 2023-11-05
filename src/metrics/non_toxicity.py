from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
import numpy as np


r_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
r_model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')


def classify_preds_non_toxicity(preds, batch_size=16):
    results = []

    for i in tqdm(range(0, len(preds), batch_size)):
        batch = r_tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)
        result = (r_model(**batch)['logits'] / 2.5).softmax(dim=1)[:,1].data.tolist()
        results.extend([1 - item for item in result])

    return np.array(results)

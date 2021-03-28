import numpy as np
import datasets
from utils import extract_pred, extract_model_pred
bert_score = datasets.load_metric('bertscore')

file = 'score_kn1_ua_bertscore.npy'
val_data = np.load('score_kn1_ua_bertscore.npy', allow_pickle=True)
if file.startswith('score'):
    pred = extract_model_pred(val_data[0])
elif file.startswith('ver'):
    pred = extract_pred(val_data[0])
truth = [x.split(':', 1)[1] for x in val_data[1]]
score = bert_score.compute(predictions=pred, references=truth, lang='en', rescale_with_baseline=True
                           )
print("Bert score F1 mean", score['f1'].mean().item())

for i in range(val_data.shape[1]):
    text = val_data[:, i]
    print(str(i) + '. Original: ', text[2])
    print(str(i) + '. Prediction: ', text[0])
    print(str(i) + '. Truth: ', text[1])


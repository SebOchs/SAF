import numpy as np
import datasets
from utils import extract_pred
bert_score = datasets.load_metric('bertscore')

val_data = np.load('final_kn1_uq_data_for_bertscore.npy', allow_pickle=True)
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


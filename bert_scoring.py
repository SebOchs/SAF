import numpy as np
import datasets
from utils import extract_pred, extract_model_pred
bert_score = datasets.load_metric('bertscore')

# Settings
########################################################################################################################
DATA = 'score_kn1_ua_bertscore.npy'
########################################################################################################################


def bert_scoring(data_path):
    val_data = np.load(data_path, allow_pickle=True)
    if data_path.startswith('score'):
        pred = extract_model_pred(val_data[0])
    elif data_path.startswith('ver'):
        pred = extract_pred(val_data[0])
    truth = [x.split(':', 1)[1] for x in val_data[1]]
    score = bert_score.compute(predictions=pred, references=truth, lang='en', rescale_with_baseline=True
                               )
    print("Bert score F1 mean", score['f1'].mean().item())
    # Uncomment to print out the model input, model prediction and gold standard for each data instance in test set
    """
    for i in range(val_data.shape[1]):
        text = val_data[:, i]
        print(str(i) + '. Original: ', text[2])
        print(str(i) + '. Prediction: ', text[0])
        print(str(i) + '. Truth: ', text[1])
    """

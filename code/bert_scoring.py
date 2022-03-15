import numpy as np
import datasets
from .utils import extract_pred, extract_model_pred
datasets.logging.set_verbosity(40)
from bert_score import score


def bert_scoring(data_path, language='en', label='score'):
    print("Bertscoring:", data_path)
    val_data = np.load(data_path, allow_pickle=True)
    if label == 'score':
        pred = extract_model_pred(val_data[0])
    elif label == 'ver':
        pred = extract_pred(val_data[0])
    truth = [x.split(':', 1)[1] for x in val_data[1]]
    if language == "en":
        bert_score = datasets.load_metric('bertscore', lang=language)
        res = bert_score.compute(predictions=pred, references=truth, lang='en', rescale_with_baseline=True)
        print("Bert score F1 mean", np.array(res['f1']).mean().item())
    elif language == "ger":
        p, r, f1 = score(pred, truth, lang="de", model_type="bert-base-multilingual-cased", rescale_with_baseline=True)
        print("Bert score F1 mean", np.array(f1).mean().item())

    # Uncomment to print out the model input, model prediction and gold standard for each data instance in test set
    """
    for i in range(val_data.shape[1]):
        text = val_data[:, i]
        print(str(i) + '. Original: ', text[2])
        print(str(i) + '. Prediction: ', text[0])
        print(str(i) + '. Truth: ', text[1])
    """

import sys
import os
import pytorch_lightning as pl
import tqdm
import math
from source_code import bert_scoring, dataloading as dl
from source_code.litT5 import LitSAFT5
from source_code.utils import *
from torch.utils.data import DataLoader, Dataset
from os.path import join
import json
import pandas as pd

# Slurm fix
sys.path.append(os.getcwd())
# Settings

########################################################################################################################
MODEL = 'models/wq_ver/wq_ver_T5_en_epoch=13-my_metric=0.3740.ckpt'
FILE = 'test.json'
########################################################################################################################


def inference(model_path, json_path, q='question', ref='reference answer', stud='student answers',
              result_path='results.json'):
    """
    Generate feedback to student answers and save to file
    :param model_path: string / path to model
    :param json_path: string / path to json file with data
    :param q: string / json key for test question
    :param ref: string / json key for reference answer
    :param stud: string / json key for student answers
    :param result_path: string / path to save location
    :return: None
    """
    with open(json_path) as f:
        data = json.load(f)
        ckpt = LitSAFT5.load_from_checkpoint(model_path)
        tokenizer = ckpt.tokenizer
        question, reference, answers = data[q], data[ref], data[stud]
        # determine mode from model path
        mode = os.path.normpath(model_path).split(os.sep)[2].split('_')
        if mode[0] == 'wq':
            with_question, label, language = True, mode[1], mode[3]
        else:
            with_question, label, language = False, mode[0], mode[2]

        # preprocess text
        texts = []
        for ans in answers:
            if language == "en":
                if with_question:
                    text = "justify and grade: question: " + question + " student: " + ans + " reference: " + reference
                else:
                    text = "justify and grade: student: " + ans + " reference: " + reference

            elif language == "ger":
                if with_question:
                    text = "Erkläre und bewerte: " + question + " Antwort: " + ans + " Lösung: " + reference
                else:
                    text = "Erkläre und bewerte: Antwort: " + ans + " Lösung: " + reference

            else:
                raise ValueError("language parameter only accepts strings \'ger\' and \'en\' for German or "
                                 "English respectively.")
            texts.append(text)

        tokenized = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

        # generate feedback
        model = ckpt.model
        model.eval()
        model.cuda()
        generated_feedback = []
        for b in tqdm.tqdm(batch(tokenized, batch_size=ckpt.batch_size),
                           total=math.ceil(len(tokenized.input_ids) / ckpt.batch_size)):
            generated_feedback += [tokenizer.decode(x, skip_special_tokens=True) for x in ckpt.model.generate(
                input_ids=b['input_ids'].cuda(), attention_mask=b['attention_mask'].cuda())]

    df = pd.DataFrame(columns=['Question', 'Reference Answer', 'Student Answer', label.capitalize(), 'Feedback'])
    if label == 'score':
        df[label.capitalize()] = [x.split()[0] for x in generated_feedback]
        df['Feedback'] = extract_model_pred(generated_feedback)
    elif label == 'ver':
        df[label.capitalize()] = extract_label(generated_feedback)
        df['Feedback'] = extract_pred(generated_feedback)
    df['Question'], df['Reference Answer'], df['Student Answer'] = question, reference, answers
    df.to_json(result_path)


if __name__ == "__main__":
    inference(MODEL, FILE)

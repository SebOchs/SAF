import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np

tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
MAX_TOKENS = 512


def save(filepath, data):
    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)
    np.save(filepath + ".npy", np.array(data), allow_pickle=True)


def preprocessing_score_kn1(path, file):
    array = []
    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            if len(ref_answers) == 1:
                for x in stud_answers:
                    response = x.find('response').text
                    feedback = x.find('response_feedback').text               
                    score = float(x.find('score').text)
                    ref = ref_answers[0].text
                    text = "justify: grade: student:" + response + tokenizer.eos_token + "reference:" + ref
                    label = str(score)
                    answer = str(score) + tokenizer.eos_token + "explanation: " + feedback
                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[
                                      :MAX_TOKENS],
                        tokenizer(answer.lower(), max_length=128, padding='max_length').input_ids[:128],
                        tokenizer(label, max_length=4, padding='max_length').input_ids                 
                    ])
    save(file, np.array(array, dtype=object))


def preprocessing_ver_kn1(path, file):
    array = []
    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            if len(ref_answers) == 1:
                for x in stud_answers:
                    response = x.find('response').text
                    feedback = x.find('response_feedback').text
                    # debug print
                    # print(file + '/' + files, response)
                    label = x.find('verification_feedback').text
                    ref = ref_answers[0].text
                    text = "justify: grade: student:" + response + tokenizer.eos_token + "reference:" + ref

                    answer = label + tokenizer.eos_token + "explanation: " + feedback
                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[:MAX_TOKENS],
                        tokenizer(answer.lower(), max_length=128, padding='max_length').input_ids[:128],
                        tokenizer(label.lower(), max_length=4, padding='max_length').input_ids,
                        len(tokenizer(answer.lower()).input_ids)
                    ])
    save(file, np.array(array, dtype=object))


# Preprocessing
preprocessing_score_kn1('kn1/training', 'preprocessed/score_kn1_train')
preprocessing_score_kn1('kn1/UA', 'preprocessed/score_kn1_ua')
preprocessing_score_kn1('kn1/UQ', 'preprocessed/score_kn1_uq')

preprocessing_ver_kn1('kn1/training', 'preprocessed/ver_kn1_train')
preprocessing_ver_kn1('kn1/UA', 'preprocessed/ver_kn1_ua')
preprocessing_ver_kn1('kn1/UQ', 'preprocessed/ver_kn1_uq')

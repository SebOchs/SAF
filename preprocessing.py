import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np
from datasets import load_dataset
import pandas
import random
import re
from collections import Counter

tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
MAX_TOKENS = 512


def save(filepath, data):
    np.save(filepath + ".npy", np.array(data), allow_pickle=True)


def preprocessing_semeval(folder_path, file_path):
    array = []
    files = os.listdir(folder_path)
    for file in files:
        root = et.parse(folder_path + '/' + file).getroot()
        for ref_answer in root[1]:
            for stud_answer in root[2]:
                text = "grade: reference: " + ref_answer.text[
                                             :-1] + tokenizer.eos_token + " student: " + stud_answer.text[:-1]
                label = stud_answer.get('accuracy')
                array.append([
                    tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                    tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                    tokenizer(label.lower(), max_length=128, padding='max_length').input_ids[:128],
                    tokenizer(label.lower(), max_length=4, padding='max_length').input_ids
                ])
    save(file_path, array)


def preprocessing_esnli(file_path, mode):
    dataset = load_dataset("esnli")[mode]
    array = []

    for i in dataset:
        text = "justify: esnli: premise: " + i['premise'] + tokenizer.eos_token + ' hypothesis: ' + i['hypothesis']
        answer = ['neutral', 'contradictory', 'entailment'][int(i['label'])] + tokenizer.eos_token + ' explanation: '
        for j in [x for x in (i['explanation_1'], i['explanation_2'], i['explanation_3']) if len(x) > 0]:
            answer += j
            array.append([
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[:MAX_TOKENS],
                tokenizer(answer.lower(), max_length=128, padding='max_length').input_ids[:128],
                tokenizer(answer.split(" explanation:", 1)[0].lower(), max_length=4, padding='max_length').input_ids
            ])
    save(file_path, array)


def preprocessing_cose(file_path, mode):
    dataset = load_dataset("cos_e", "v1.11")[mode]
    array = []

    for i in dataset:
        text = "justify: cose: question: " + i['question'] + tokenizer.eos_token + \
               ' '.join(' choice: ' + x + tokenizer.eos_token for x in i['choices'])
        answer = i['answer'] + tokenizer.eos_token + ' explanation: ' + i['abstractive_explanation']
        array.append([
            tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
            tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[:MAX_TOKENS],
            tokenizer(answer.lower(), max_length=128, padding='max_length').input_ids[:128],
            tokenizer(answer.split(" explanation:", 1)[0].lower(), max_length=4,
                               padding='max_length').input_ids
         ])
    save(file_path, array)


def preprocessing_glucose(folder_path, file_path):
    def remove_relation(specific_answer):
        idx = [m.start() for m in re.finditer('>', specific_answer)]
        assert len(idx) == 2
        return specific_answer[:idx[0]] + '.' + specific_answer[idx[1] + 1:]

    dataset = pandas.read_csv(folder_path)
    array = []
    for i in range(len(dataset)):
        date = dataset.iloc[i]
        text = "justify: glucose: " + date.story
        usable_answers = [date[x] for x in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44] if date[x] != 'escaped']
        for j in usable_answers:
            answer = "explanation: " + remove_relation(j)
            array.append([
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[:MAX_TOKENS],
                tokenizer(answer.lower(), max_length=128, padding='max_length').input_ids[:128],
                tokenizer('', max_length=4, padding='max_length').input_ids
            ])
    # control_array = random.choices(array, k=10)
    random.shuffle(array)
    split = round(len(array) * 0.8)
    save(file_path + '_train', array[:split])
    save(file_path + '_test', array[split:])


def preprocessing_kn1(path, file):
    array = []
    test = []
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
                        tokenizer(label, max_length=4, padding='max_length').input_ids,
                        # len(tokenizer(text.lower()).input_ids)
                    ])
    # x = Counter([float(x[4]) for x in array])

    save(file, array)


def preprocessing_asag_kn1(path, file):
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
    # x = min([x[4] for x in array])

    save(file, array)


# Preprocess for Multitasking
# preprocessing_glucose('datasets/raw/GLUCOSE_training_data_final.csv', 'datasets/preprocessed/glucose')
preprocessing_cose('datasets/preprocessed/cose_train', 'train')
# preprocessing_cose('datasets/preprocessed/cose_test', 'validation')
preprocessing_esnli("datasets/preprocessed/esnli_train", 'train')
# preprocessing_esnli("datasets/preprocessed/esnli_val", 'validation')
# preprocessing_esnli("datasets/preprocessed/esnli_test", 'test')


preprocessing_semeval("datasets/raw/sciEntsBank_training", "datasets/preprocessed/seb_train")
preprocessing_semeval("datasets/raw/sciEntsBank_testing/test-unseen-answers", "datasets/preprocessed"
                                                                             "/seb_test_ua")
preprocessing_semeval("datasets/raw/sciEntsBank_testing/test-unseen-domains", "datasets/preprocessed"
                                                                              "/seb_test_ud")
preprocessing_semeval("datasets/raw/sciEntsBank_testing/test-unseen-questions", "datasets/preprocessed"
                                                                         "/seb_test_uq")
preprocessing_asag_kn1('datasets/raw/kn1/training', 'datasets/preprocessed/asag_kn1_train')
preprocessing_asag_kn1('datasets/raw/kn1/UA', 'datasets/preprocessed/asag_kn1_ua')
preprocessing_asag_kn1('datasets/raw/kn1/UQ', 'datasets/preprocessed/asag_kn1_uq')
import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np
import sys
from utils import *
from datasets import load_dataset
# Settings
########################################################################################################################
# tokenizer and max length
TOKENIZER = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
MAX_TOKENS = 512
OUTPUT_LENGTH = 128
# paths to kn1 data set folders
TRAIN = 'kn1/training'
UA = 'kn1/UA'
UQ = 'kn1/UQ'
########################################################################################################################


def preprocessing_score_kn1(path, file, tokenizer):
    """
    Preprocessor for KN1 data set with grading scores as labels
    :param tokenizer: huggingface tokenizer to preprocess the data
    :param path: string - path to the folder containing the raw data
    :param file: string - file path, where to save the preprocessed data
    :return: Nothing
    """
    array = []
    # Iterate over files in folder
    for files in os.listdir(path):
        if files.endswith('.xml'):
            root = et.parse(path + '/' + files).getroot()
            # get reference amd student answers from the files
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            # make sure only one reference answer is given
            if len(ref_answers) == 1:
                for x in stud_answers:
                    # get student answer, feedback and score from file
                    response = x.find('response').text
                    feedback = x.find('response_feedback').text
                    score = str(float(x.find('score').text))
                    ref = ref_answers[0].text
                    # prepare input for T5 model
                    text = "justify: grade: student:" + response + tokenizer.eos_token + "reference:" + ref
                    # prepare output
                    answer = score + tokenizer.eos_token + "explanation: " + feedback

                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True)
                            .attention_mask,
                        tokenizer(answer.lower(), max_length=OUTPUT_LENGTH, padding='max_length', truncation=True)
                            .input_ids,
                        # max length of score is 4
                        tokenizer(score, max_length=4, padding='max_length', truncation=True).input_ids
                    ])
            else:
                raise ValueError("Multiple reference answers were found in file " + path + '/' + files)
    save(file, np.array(array, dtype=object))


def preprocessing_ver_kn1(path, file, tokenizer):
    """
    Preprocessor for KN1 data set with verification feedback as labels
    :param tokenizer: huggingface tokenizer to preprocess the data
    :param path: string - path to the folder containing the raw data
    :param file: string - file path, where to save the preprocessed data
    :return: Nothing
    """
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
                    label = x.find('verification_feedback').text
                    ref = ref_answers[0].text
                    text = "justify: grade: student:" + response + tokenizer.eos_token + "reference:" + ref

                    answer = label + tokenizer.eos_token + "explanation: " + feedback
                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True)
                            .attention_mask,
                        tokenizer(answer.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                        tokenizer(label.lower(), max_length=4, padding='max_length', truncation=True).input_ids,
                        len(tokenizer(answer.lower()).input_ids)
                    ])
    save(file, np.array(array, dtype=object))


def preprocessing_label_only_kn1(path, file, tokenizer):
    """
        Preprocessor for KN1 data set with verification feedback as labels
        :param tokenizer: huggingface tokenizer to preprocess the data
        :param path: string - path to the folder containing the raw data
        :param file: string - file path, where to save the preprocessed data
        :return: Nothing
        """
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
                    label = x.find('verification_feedback').text
                    ref = ref_answers[0].text
                    text = "justify: grade: student:" + response + tokenizer.eos_token + "reference:" + ref

                    answer = label + tokenizer.eos_token + "explanation: " + feedback
                    array.append([
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True)
                            .attention_mask,
                        tokenizer(label.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                        tokenizer(label.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                        len(tokenizer(answer.lower()).input_ids)
                    ])
    save(file, np.array(array, dtype=object))


def preprocessing_semeval(folder_path, file_path, tokenizer):
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
                    tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                    tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                    tokenizer(label.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                    tokenizer(label.lower(), max_length=4, padding='max_length', truncation=True).input_ids
                ])
    save(file_path, array)


def preprocessing_esnli(file_path, mode, tokenizer):
    dataset = load_dataset("esnli")[mode]
    array = []

    for i in dataset:
        text = "justify: esnli: premise: " + i['premise'] + tokenizer.eos_token + ' hypothesis: ' + i['hypothesis']
        answer = ['neutral', 'contradictory', 'entailment'][int(i['label'])] + tokenizer.eos_token + ' explanation: '
        for j in [x for x in (i['explanation_1'], i['explanation_2'], i['explanation_3']) if len(x) > 0]:
            answer += j
            array.append([
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
                tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).attention_mask,
                tokenizer(answer.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
                tokenizer(answer.split(" explanation:", 1)[0].lower(), max_length=4, padding='max_length',
                          truncation=True).input_ids
            ])
    save(file_path, array)


def preprocessing_cose(file_path, mode, tokenizer):
    dataset = load_dataset("cos_e", "v1.11")[mode]
    array = []

    for i in dataset:
        text = "justify: cose: question: " + i['question'] + tokenizer.eos_token + \
               ' '.join(' choice: ' + x + tokenizer.eos_token for x in i['choices'])
        answer = i['answer'] + tokenizer.eos_token + ' explanation: ' + i['abstractive_explanation']
        array.append([
            tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).input_ids,
            tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length', truncation=True).attention_mask,
            tokenizer(answer.lower(), max_length=128, padding='max_length', truncation=True).input_ids,
            tokenizer(answer.split(" explanation:", 1)[0].lower(), max_length=4, padding='max_length',
                      truncation=True).input_ids
         ])
    save(file_path, array)


# Preprocessing
# For scores as labels
"""
preprocessing_score_kn1(TRAIN, 'preprocessed/score_kn1_train', TOKENIZER)
preprocessing_score_kn1(UA, 'preprocessed/score_kn1_ua', TOKENIZER)
preprocessing_score_kn1(UQ, 'preprocessed/score_kn1_uq', TOKENIZER)
# For verification feedback as labels
preprocessing_ver_kn1(TRAIN, 'preprocessed/ver_kn1_train', TOKENIZER)
preprocessing_ver_kn1(UA, 'preprocessed/ver_kn1_ua', TOKENIZER)
preprocessing_ver_kn1(UQ, 'preprocessed/ver_kn1_uq', TOKENIZER)
# For Multi
# Label only
"""
preprocessing_label_only_kn1(TRAIN, 'preprocessed/label_only_kn1', TOKENIZER)
# seb, esnli and cose
# preprocessing_semeval('sciEntsBank_training', 'preprocessed/seb_train', TOKENIZER)
# preprocessing_esnli('preprocessed/esnli_train', 'train', TOKENIZER)
# preprocessing_cose('preprocessed/cose_train', 'train', TOKENIZER)
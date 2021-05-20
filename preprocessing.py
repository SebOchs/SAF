import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np

# Settings
########################################################################################################################
# tokenizer and max length
TOKENIZER = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
MAX_TOKENS = 512
OUTPUT_LENGTH = 128
# paths to kn1 data set
TRAIN = 'kn1/training'
UA = 'kn1/UA'
UQ = 'kn1/UQ'
########################################################################################################################


def save(filepath, data):
    """
    Function to save the preprocessed data into a folder structure
    :param filepath: string - path of the file to save
    :param data: list of preprocessed data
    :return: Nothing
    """
    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)
    np.save(filepath + ".npy", np.array(data), allow_pickle=True)


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
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[
                        :MAX_TOKENS],
                        tokenizer(answer.lower(), max_length=OUTPUT_LENGTH, padding='max_length').input_ids[:128],
                        # max length of score is 4
                        tokenizer(score, max_length=4, padding='max_length').input_ids
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
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').input_ids[:MAX_TOKENS],
                        tokenizer(text.lower(), max_length=MAX_TOKENS, padding='max_length').attention_mask[
                            :MAX_TOKENS],
                        tokenizer(answer.lower(), max_length=128, padding='max_length').input_ids[:128],
                        tokenizer(label.lower(), max_length=4, padding='max_length').input_ids,
                        len(tokenizer(answer.lower()).input_ids)
                    ])
    save(file, np.array(array, dtype=object))


# Preprocessing
# For scores as labels
preprocessing_score_kn1(TRAIN, 'preprocessed/score_kn1_train', TOKENIZER)
preprocessing_score_kn1(UA, 'preprocessed/score_kn1_ua', TOKENIZER)
preprocessing_score_kn1(UQ, 'preprocessed/score_kn1_uq', TOKENIZER)
# For verification feedback as labels
preprocessing_ver_kn1(TRAIN, 'preprocessed/ver_kn1_train', TOKENIZER)
preprocessing_ver_kn1(UA, 'preprocessed/ver_kn1_ua', TOKENIZER)
preprocessing_ver_kn1(UQ, 'preprocessed/ver_kn1_uq', TOKENIZER)

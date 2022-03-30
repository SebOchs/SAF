import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import os
import sys

# Slurm fix
sys.path.append(os.getcwd())


def split(number, portion=0.9):
    """
    splitting a data set into train and val set
    :param number: Int / number of samples in dataset
    :param portion: Float / percentile of samples that go into train set
    :return: list of Int / numbers indicating samples needed in train and val set according to portion
    """
    # splitting data set
    return [round(portion * number), round((1 - portion) * number)]


def isfloat(value):
    """
    test if string can be converted to float
    :param value: String
    :return: boolean
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def mse(pred, labs):
    """
    Calculates MSE
    :param pred: sequence of Strings / predicted score values as strings
    :param labs: sequence of Strings / true score values as strings
    :return: (Int, Int) / MSE of valid samples AND number of invalid samples
    """
    idx = np.where(np.array([isfloat(x) for x in pred]) == True)
    if idx[0].size > 0:
        preds = np.array([float(x) for x in pred[idx]])
        lab = np.array([float(x) for x in labs[idx]])
        if lab.size - idx[0].size > 0:
            print('\nInvalid validation examples: ', labs.size - idx[0].size)
        mse_val = mean_squared_error(lab, preds)
        out_of_bound = np.sum(preds > 1) + np.sum(preds < 0)
        if out_of_bound > 0:
            print("\nMSE out of bound values: ", out_of_bound)
        if mse_val > 1:
            print('MSE greater than 1')
            return 1, labs.size - idx[0].size
        else:
            return mse_val, labs.size - idx[0].size

    else:
        print('\nInvalid validation')
        return 1, labs.size - idx[0].size


def extract_pred(predictions):
    """
    extract the model prediction without the label at the beginning
    :param predictions: list of Strings / complete predicted output
    :return: list of Strings / predicted output without labels
    """
    array = []
    for pred in predictions:
        try:
            x = pred.split(':', 1)[1]
        except IndexError:
            try:
                if pred.startswith('partially correct'):
                    x = pred.split(' ', 1)[2]
                else:
                    x = pred.split(' ', 1)[1]
            except IndexError:
                x = pred
        array.append(x)
    return array


def extract_model_pred(predictions):
    """
    extract the model prediction without the label at the beginning
    :param predictions: list of Strings / complete predicted output
    :return: list of Strings / predicted output without labels
    """
    #
    array = []
    for pred in predictions:
        try:
            x = pred.split('explanation:', 1)[1]
        except IndexError:
            try:
                x = pred.split(':', 1)[1]
            except IndexError:
                x = pred
        array.append(x)
    return array


def extract_label(predictions):
    """
    extract the predicted label without the following prediction
    :param predictions: list of Strings / complete predicted output
    :return: list of Strings / only labels
    """
    # extract the predicted label without the following prediction
    array = []
    for pred in predictions:
        if pred.startswith('correct'):
            x = 'correct'
        elif pred.startswith('incorrect'):
            x = 'incorrect'
        elif pred.startswith('partially correct'):
            x = 'partially correct'
        else:
            x = 'wrong label'
        array.append(x)
    return array


def save(filepath, data):
    """
    Function to save the preprocessed data into a folder structure
    :param filepath: string - path of the file to save
    :param data: list of preprocessed data
    :return: Nothing
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, np.array(data, dtype=object), allow_pickle=True)


def to_list(x):
    """
    Function to convert a list of list of elements to list of elements
    :param x: list of list of objects
    :return: list of objects
    """
    return [a for b in x for a in b]

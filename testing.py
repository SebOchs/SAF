import sys
import os
import pytorch_lightning as pl
from source_code import bert_scoring, dataloading as dl
from source_code.litT5 import LitSAFT5
from torch.utils.data import DataLoader
from os.path import join

# Slurm fix
sys.path.append(os.getcwd())
# Settings

########################################################################################################################
MODEL = 'models/wq_score/wq_score_mT5_ger_epoch=17-my_metric=0.1950.ckpt'
UA = True
UQ = True
BERT_SCORE = True
########################################################################################################################


def testing(model_path, ua=True, uq=False, bert_score=False, mode=None):
    # get mode from model
    if mode:
        with_question, label, language = mode
    else:
        mode = os.path.normpath(model_path).split(os.sep)[2].split('_')
        if mode[0] == 'wq':
            with_question, label, language = True, mode[1], mode[3]
        else:
            with_question, label, language = False, mode[0], mode[2]
    mode = '_'.join(['wq', label]) if with_question else label

    # get preprocessed test sets
    if language == 'en':
        folder = join('preprocessed', 'english')
    elif language == 'ger':
        folder = join('preprocessed', 'german')
    else:
        raise ValueError("Unsupported language or string")
    test_set_paths = []
    if ua:
        test_set_paths.append(join(folder, mode + '_ua.npy'))
    if uq:
        test_set_paths.append(join(folder, mode + '_uq.npy'))

    # Load model

    trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=10)
    for i in test_set_paths:
        test_set = i.rsplit('_', 1)[1][:2]
        print("Testing on", test_set.upper())
        test_model = LitSAFT5.load_from_checkpoint(model_path, test=test_set, bert_scoring=bert_score)
        test_loader = DataLoader(dl.T5Dataset(i))
        trainer.test(test_model, test_dataloaders=test_loader, verbose=True)
        if bert_score:
            bert_scoring.bert_scoring(join('models', mode, language + '_' + i.rsplit('_', 1)[1]), language=language,
                                      label=label)


if __name__ == "__main__":
    testing(MODEL, ua=UA, uq=UQ, bert_score=BERT_SCORE)

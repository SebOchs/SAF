import sys
import os
import pytorch_lightning as pl
import bert_scoring
from litT5 import LitScoreFineT5, LitVerFineT5
from torch.utils.data import DataLoader
import dataloading as dl
# Slurm fix
sys.path.append(os.getcwd())
# Settings
########################################################################################################################
MODE = "wq_score_jobber"
DATASET = "jobber"
LANG = "others"
MODEL = "models/"+ MODE +"/wq_ver_jobber_mt5_epoch=35-my_metric=0.6245_wq_ver_jobber.ckpt"
TEST_SET = 'preprocessed/'+ MODE + '_UQ.npy'
SECOND_TEST_SET = 'preprocessed/'+ MODE + '_UA.npy'
BERT_DATA = 'models/' + MODE + '/' + DATASET + '_uq_bertscore.npy'
BERT_SECOND_DATA = 'models/' + MODE + '/' + DATASET + '_ua_bertscore.npy'
########################################################################################################################

def testing(model_path, test_set, mode="score_jobber"):
    # Load test set from file path test_set
    test_loader = DataLoader(dl.T5Dataset(test_set))

    # Load model from checkpoint file path
    if MODE.split('_')[1] == 'ver' or MODE.split('_')[0] == 'ver':
        t5_test = LitVerFineT5.load_from_checkpoint(model_path)
    else:
        t5_test = LitScoreFineT5.load_from_checkpoint(model_path, mode=mode)

    trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=0)
    trainer.test(t5_test, test_dataloaders=test_loader, verbose=True) #, show_progress_bar=False)


if __name__ == "__main__":
    """
    print("Testing model ", MODEL)
    print("Testing on UQ")
    testing(MODEL, TEST_SET)
    print("Testing on UA")
    testing(MODEL, SECOND_TEST_SET)
    """
    print("BERTscoring UQ")
    bert_scoring.bert_scoring(BERT_DATA, lang=LANG)
    print("BERTscoring UA")
    bert_scoring.bert_scoring(BERT_SECOND_DATA, lang=LANG)


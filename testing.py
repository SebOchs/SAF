import sys
import os
import pytorch_lightning as pl
from litT5 import LitScoreFineT5, LitVerFineT5
from torch.utils.data import DataLoader
import dataloading as dl
# Slurm fix
sys.path.append(os.getcwd())
# Settings
########################################################################################################################
MODEL = "models/ver/kn1_t5_epoch=0-my_metric=0.0000_ver.ckpt"
TEST_SET = 'preprocessed/ver_kn1_ua.npy'
########################################################################################################################


def testing(model_path, test_set):
    # Load test set from file path test_set
    test_loader = DataLoader(dl.T5Dataset(test_set))

    # Load model from checkpoint file path
    if model_path.split('/')[1] == 'score':
        t5_test = LitScoreFineT5.load_from_checkpoint(model_path)
    elif model_path.split('/')[1] == 'ver':
        t5_test = LitVerFineT5.load_from_checkpoint(model_path)

    trainer = pl.Trainer(gpus=1)
    trainer.test(t5_test, test_dataloaders=test_loader)


testing(MODEL, TEST_SET)


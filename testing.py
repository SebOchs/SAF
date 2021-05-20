import pytorch_lightning as pl
from litT5 import LitScoreFineT5, LitVerFineT5
from torch.utils.data import DataLoader
import dataloading as dl

# Settings
########################################################################################################################
MODEL = "models/kn1_t5_epoch=1-my_metric=0.3404.ckpt"
TEST_SET = 'preprocessed/score_kn1_uq.npy'
MODE = 'score'  # or 'ver'
########################################################################################################################


def testing(model_path, test_set, mode):
    # Load test set from file path test_set
    test_loader = DataLoader(dl.T5Dataset(test_set))

    # Load model from checkpoint file path
    if mode == 'score':
        t5_test = LitScoreFineT5.load_from_checkpoint(model_path)
    elif mode == 'ver':
        t5_test = LitVerFineT5.load_from_checkpoint(model_path)

    trainer = pl.Trainer(gpus=1)
    trainer.test(t5_test, test_dataloaders=test_loader)


testing(MODEL, TEST_SET, MODE)


import pytorch_lightning as pl
from litT5 import LitScoreFineT5, LitVerFineT5
from torch.utils.data import DataLoader
import dataloading as dl

# If test data set is not unseen-answers, specify here:
# test_loader = DataLoader(dl.T5Dataset('preprocessed/score_kn1_uq.npy'))

# Load model (specify if model is either LitScoreFineT5 or LitVerFineT5)
t5_test = LitScoreFineT5.load_from_checkpoint("models/kn1_t5_epoch=1-my_metric=0.3404.ckpt")
trainer = pl.Trainer(gpus=1)

# remove test_dataloaders if results for default test data set are needed
trainer.test(t5_test#, test_dataloaders=test_loader
)
print("finished testing")

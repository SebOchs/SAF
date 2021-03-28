import pytorch_lightning as pl
from litT5 import LitFineT5, LitAsagFineT5
from torch.utils.data import DataLoader
import dataloading as dl

# If test data set is not unseen-answers, specify here:
test_loader = DataLoader(dl.T5Dataset('datasets/preprocessed/asag_kn1_uq.npy'))

# Load model (specify if model is either LitFineT5 or LitAsagFineT5)
t5_test = LitAsagFineT5.load_from_checkpoint("models/final_kn1_t5_epoch=35-my_metric=0.3344.ckpt")
trainer = pl.Trainer(gpus=1)

# remove test_dataloaders if results for default test data set are needed
trainer.test(t5_test, test_dataloaders=test_loader)
print("finished testing")

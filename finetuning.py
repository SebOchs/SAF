import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from litT5 import LitScoreFineT5, LitVerFineT5, LitMultiT5
# Slurm fix
sys.path.append(os.getcwd())

# Settings
########################################################################################################################
MODE = ['ver', 'score', 'multi']  # Replace with 'ver' for finetuning on verification feedback
# Hyperparameters
BATCH_SIZE = 2
EPOCH = 64
ACCUMULATE_GRAD = 4  # best performing
# Training settings
N_TOP_MODELS = 3
DISTRIBUTED = True
N_GPUS = 2
# Path to model ckpt if finetuning the multitask model
multi_path = ''
########################################################################################################################


def finetuning(mode, batch_size=4, epochs=64, acc_grad=8, top_k=3, ddp=False, gpus=1, ckpt=None):
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/' + mode,
        monitor='my_metric',
        mode="max",
        filename='kn1_t5_{epoch}-{my_metric:.4f}_' + mode,
        save_top_k=top_k
    )
    # Early Stopping
    early = EarlyStopping(
        monitor='my_metric',
        mode="max",
        patience=3,
        verbose=False
    )
    # Initialize model and trainer
    if mode == 'score':
        t5_version = LitScoreFineT5(batch_size)
    elif mode == 'ver':
        t5_version = LitVerFineT5(batch_size)
    elif mode == 'multi':
        model = LitMultiT5.load_from_checkpoint(ckpt).model
        t5_version = LitVerFineT5(batch_size, model=model)

    if ddp:
        trainer = pl.Trainer(
            gpus=gpus,
            auto_select_gpus=True,
            num_nodes=1,
            accelerator='ddp',
            max_epochs=epochs,
            accumulate_grad_batches=acc_grad,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback, early],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=100
        )
    else:
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=epochs,
            accumulate_grad_batches=acc_grad,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback, early],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=100
        )
    trainer.fit(t5_version)
    print("Best model with batchsize {} and acc_grad {} is: ".format(batch_size, acc_grad) +
          checkpoint_callback.best_model_path)

# Example scripts

# finetuning('ver', batch_size=BATCH_SIZE, epochs=EPOCH, acc_grad=ACCUMULATE_GRAD, gpus=N_GPUS)
# finetuning('score', batch_size=BATCH_SIZE, epochs=EPOCH, acc_grad=ACCUMULATE_GRAD, gpus=N_GPUS)



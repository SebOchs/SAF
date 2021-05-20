import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from litT5 import LitScoreFineT5, LitVerFineT5

# Settings
########################################################################################################################
MODE = 'score'  # Replace with 'ver' for finetuning on verification feedback
# Hyperparameters
BATCH_SIZE = 4
EPOCH = 64
ACCUMULATE_GRAD = 2
# Training settings
N_TOP_MODELS = 3
DISTRIBUTED = True
N_GPUS = 2
########################################################################################################################


def finetuning(mode):
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='my_metric',
        mode="max",
        filepath='models/score_kn1_t5_{epoch}-{my_metric:.4f}',
        save_top_k=N_TOP_MODELS
    )
    # Initialize model and trainer
    if mode == 'score':
        t5_version = LitScoreFineT5(BATCH_SIZE)
    elif mode == 'ver':
        t5_version = LitVerFineT5(BATCH_SIZE)

    if DISTRIBUTED:
        trainer = pl.Trainer(
            gpus=N_GPUS,
            num_nodes=1,
            accelerator='ddp',
            max_epochs=EPOCH,
            accumulate_grad_batches=ACCUMULATE_GRAD,
            checkpoint_callback=checkpoint_callback,
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=10
        )
    else:
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=EPOCH,
            accumulate_grad_batches=ACCUMULATE_GRAD,
            checkpoint_callback=checkpoint_callback,
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=10
        )

    trainer.fit(t5_version)


finetuning('score')

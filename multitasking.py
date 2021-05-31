import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from litT5 import LitMultiT5

# Slurm fix
sys.path.append(os.getcwd())


def multitasking(batch_size, acc_grad, ddp=False, gpus=1):
    # Init model
    t5_version = LitMultiT5(batch_size)
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/multitask',
        monitor='my_metric',
        mode="max",
        filename='kn1_t5_{epoch}-{my_metric:.4f}_multitask',
        save_top_k=3
    )
    if ddp:
        trainer = pl.Trainer(
            gpus=gpus,
            auto_select_gpus=True,
            num_nodes=1,
            accelerator='ddp',
            max_epochs=64,
            accumulate_grad_batches=acc_grad,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=100,
            reload_dataloaders_every_epoch=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=64,
            accumulate_grad_batches=acc_grad,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=100,
            reload_dataloaders_every_epoch=True,
        )
    trainer.fit(t5_version)


multitasking(2, 8)

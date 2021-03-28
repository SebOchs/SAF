import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from litT5 import LitVerFineT5

checkpoint_callback = ModelCheckpoint(
    monitor='my_metric',
    mode="max",
    filepath='models/ver_kn1_t5_{epoch}-{my_metric:.4f}',
    save_top_k=3
)
ver_t5 = LitVerFineT5(2)
trainer = pl.Trainer(
    gpus=2,
    num_nodes=1,
    accelerator='ddp',
    max_epochs=64,
    accumulate_grad_batches=2,
    checkpoint_callback=checkpoint_callback,
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=10
)

trainer.fit(ver_t5)

print("finished fine-tuning")

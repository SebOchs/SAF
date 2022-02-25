import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from litT5 import LitScoreFineT5, LitVerFineT5, LitMultiT5
from torch.cuda import empty_cache, memory_allocated, memory_cached, current_device


# Slurm fix
sys.path.append(os.getcwd())

# Settings
########################################################################################################################
MODE = "wq_ver_jobber"  # Replace with 'ver' for finetuning on verification feedback
MODEL = "mt5"
# Hyperparameters
# BATCH_SIZE = [2, 4] # 2 is best for score, 4 is best for ver
EPOCH = 80
ACCUMULATE_GRAD = 10 # best performing was 6 for kn1, 8 for job_wq
BATCH_SIZE = 2
# Training settings
N_TOP_MODELS = 1
PATIENCE = 5
DISTRIBUTED = False
N_GPUS = 1
# Path to model ckpt if finetuning the multitask model
multi_path = ''
PRECISION = 32


########################################################################################################################


def finetuning(mode, batch_size=4, epochs=64, acc_grad=8, top_k=3, ddp=False, gpus=1, ckpt=None, precision=16, model_version="mT5"):
    # Checkpointing
    """
    print("Finetuning has started. Getting memory info:")
    print(memory_cached(current_device()))
    print(memory_allocated(current_device()))
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/' + mode,
        monitor='my_metric',
        mode="max",
        filename=mode+ '_' + model_version +'_{epoch}-{my_metric:.4f}_' + mode,
        save_top_k=top_k
    )
    # Early Stopping
    early = EarlyStopping(
        monitor='my_metric',
        mode="max",
        patience=PATIENCE,
        verbose=False
    )
    # Initialize model and trainer
    if mode.split("_")[1] == 'ver' or mode.split("_")[0] == 'ver':
        t5_version = LitVerFineT5(batch_size, mode=MODE, model=MODEL)
    elif mode == 'multi':
        model = LitMultiT5.load_from_checkpoint(ckpt).model
        t5_version = LitVerFineT5(batch_size, model=model)
    else:
        t5_version = LitScoreFineT5(batch_size, mode=mode, model_version=model_version)

    if ddp:
        trainer = pl.Trainer(
            gpus=gpus,
            auto_select_gpus=True,
            num_nodes=1,
            accelerator='ddp',
            precision=precision,
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
            precision=precision,
            #amp_backend="apex",
            #amp_level="O3",
            log_gpu_memory="all",
            accumulate_grad_batches=acc_grad,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback, early],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=100,
            # stochastic_weight_avg=False
        )
    """
    print("Before fit")
    print(memory_cached(current_device()))
    print(memory_allocated(current_device()))
    """
    trainer.fit(t5_version)
    print("Best model with batchsize {} and acc_grad {} is: ".format(batch_size, acc_grad) +
          checkpoint_callback.best_model_path)


# Example scripts
if __name__ == "__main__":
    #finetuning('score', batch_size=2, epochs=EPOCH, acc_grad=ACCUMULATE_GRAD, gpus=N_GPUS)
    print("batch size = {}, acummulated gradients = {}, epochs = {}, patience = {}, topk= {}, precision = {}".format(BATCH_SIZE, ACCUMULATE_GRAD, EPOCH, PATIENCE, N_TOP_MODELS, PRECISION))
    finetuning(MODE, batch_size=BATCH_SIZE,top_k=N_TOP_MODELS, epochs=EPOCH, acc_grad=ACCUMULATE_GRAD, gpus=N_GPUS, precision=PRECISION, model_version=MODEL)



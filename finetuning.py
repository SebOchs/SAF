from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from code.litT5 import *
# Slurm fix
sys.path.append(os.getcwd())

# Settings
########################################################################################################################
LANG = 'en' # 'en' or 'ger'
QUESTION = True # train with or w/o questions
LABEL = 'ver_saf' # 'ver' or 'score'
# Hyperparameters
BATCH_SIZE = 2 # 2 is best for score, 4 is best for ver
EPOCH = 80
ACCUMULATE_GRAD = 6 # best performing was 6 for en, 8 for ger
# Training settings
N_TOP_MODELS = 1
PATIENCE = 4
DISTRIBUTED = False
N_GPUS = 0
PRECISION = 16
########################################################################################################################
if LANG == 'ger':
    MODEL = 'mT5'
elif LANG == 'en':
    MODEL = 'T5'


def finetuning(language, with_questions, label, batch_size=4, epochs=64, acc_grad=8, top_k=3, ddp=False, gpus=1,
               precision=16, model_version="mT5"):
    """
    Finetuning function to set up model training
    :param language: String / indicates language
    :param with_questions: Boolean / train with or w/o questions in the text sequences
    :param label: String / train with 'score' s or 'ver' ification feedback
    :param batch_size: Int
    :param epochs: Int
    :param acc_grad: Int
    :param top_k: Int
    :param ddp: Boolean
    :param gpus: Int
    :param precision: Int
    :param model_version: String / abbreviation of the model that is finetuned
    :return:
    """
    if with_questions:
        mode = 'wq_' + label
    else:
        mode = label

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/' + mode,
        monitor='my_metric',
        mode="max",
        filename= mode + '_' + model_version + '_' + language + '_{epoch}-{my_metric:.4f}',
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
    model = LitSAFT5(batch_size, with_questions=with_questions, label=label, language=language)

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
            accumulate_grad_batches=acc_grad,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback, early],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=10
        )

    trainer.fit(model)
    print("Best model with batchsize {} and acc_grad {} is: ".format(batch_size, acc_grad) +
          checkpoint_callback.best_model_path)


# Example scripts
if __name__ == "__main__":
    print("batch size = {}, acummulated gradients = {}, epochs = {}, patience = {}, topk= {}, precision = {}".format(
        BATCH_SIZE, ACCUMULATE_GRAD, EPOCH, PATIENCE, N_TOP_MODELS, PRECISION))

    finetuning(LANG, QUESTION, LABEL, batch_size=BATCH_SIZE,top_k=N_TOP_MODELS, epochs=EPOCH, acc_grad=ACCUMULATE_GRAD,
               gpus=N_GPUS, precision=PRECISION, model_version=MODEL)


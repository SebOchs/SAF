import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer,AutoModelForSeq2SeqLM, AutoTokenizer
from utils import *
import dataloading as dl
import warnings
import datasets
import torch
import math
import numpy as np
from datetime import datetime

from torch.cuda import empty_cache, memory_allocated, memory_cached, current_device


# set seeds
pl.seed_everything(np.random.randint(0, 1000))
# just here for cleaner console output
warnings.filterwarnings("ignore")

# text quality metrics
sacrebleu = datasets.load_metric('sacrebleu')
rouge = datasets.load_metric('rouge')
meteor = datasets.load_metric('meteor')


# pytorch-lightning module to fine-tune model on scores
class LitScoreFineT5(pl.LightningModule):
    """datasets can be either:
        - score_jobber
        - wq_score_jobber
        - wq_score_kn1
        - score_kn1
        - or a prefix selected during preprocessing """
    def __init__(self, batch_size, mode, model_version="mt5"):
        super(LitScoreFineT5, self).__init__()
        # Load model and tokenizer
        if model_version == "mt5":
            print("Using mt5 Model")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        else:
            print("Using T5 Model")
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        print("Training on the following datasets: ", mode)
        data = dl.T5Dataset('preprocessed/' + mode +'_train.npy')
        self.test_data = dl.T5Dataset('preprocessed/' + mode +'_UA.npy')

        self.batch_size = batch_size
        # Load and split data
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.save_hyperparameters()
        """
        print("After Init of Model")
        print(memory_cached(current_device()))
        print(memory_allocated(current_device()))
        """

    def forward(self, tok_seq, attn_seq):
        # force min length of prediction
        """
        print("Forward Pass")
        print(memory_cached(current_device()))
        print(memory_allocated(current_device()))
        """
        state = self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=11,
                                                         max_length=128)[0], skip_special_tokens=True)
        return state

    def training_step(self, batch, batch_idx):
        """
        print("Training Step", self.device)
        print(memory_cached(current_device()))
        print(memory_allocated(current_device()))
        empty_cache()
        """
        text, text_attn, answer, lab = batch
        loss = self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()
        #self.log('Train-Cross-Entropy-Loss', float(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        print("Val Step")
        print(memory_cached(current_device()))
        print(memory_allocated(current_device()))
        """
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):
        # validation array: first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                    [x['label'] for x in outputs], [x['prediction'].split(' ', 1)[0] for x in outputs]]

        pred = extract_model_pred(val_data[0])
        #print("Valdata at 1: ",val_data[1])
        truth = [x.split(' ', 2)[2] for x in val_data[1]]

        # calculate model selection metrics
        acc_data = np.array(val_data[2:])
        for i, point in enumerate(val_data[0]):
            print("Predicted: ", pred[i], "Gold: ", truth[i])
            print("Predicted Label: ", acc_data[1][i], "Gold label: ", acc_data[0][i])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']
        if len(acc_data[1]) > 0:
            mse_val, invalid = mse(acc_data[1], acc_data[0])
            self.log('my_metric', (sacrebleu_score / 100 + rouge_score + meteor_score) / 3 * (1 - mse_val) *
                     (1 - invalid / len(acc_data[1])))
        else:
            print('\nInvalid mse')
            mse_val, invalid = 1
            self.log('my_metric', 0)

        self.log('sacreBleu', sacrebleu_score)
        self.log('ROUGE', rouge_score)
        self.log('METEOR', meteor_score)
        print('MSE = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(mse_val, sacrebleu_score, rouge_score, meteor_score))

    def test_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        #print("Test stepping")
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def test_epoch_end(self, outputs):
        test_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                     [x['original'] for x in outputs],
                     [x['label'] for x in outputs], [x['prediction'].split(' ', 1)[0] for x in outputs]]

        pred = extract_model_pred(test_data[0])
        truth = [x.split(' ', 2)[2] for x in test_data[1]]

        # calculate model metrics
        acc_data = np.array(test_data[3:])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']

        # Check if there are no empty predictions for mse
        if len(acc_data[1]) > 0:
            mse_val, invalid = mse(acc_data[1], acc_data[0])
            self.log('mse', mse_val)
        else:
            print('\nInvalid mse')
            self.log('mse', 0)

        self.log('bleu', sacrebleu_score)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        print('MSE = {:.4f}, RMSE = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(mse_val, math.sqrt(mse_val), sacrebleu_score, rouge_score, meteor_score))

        # FIXME: Currently recognizes different test sets by length, not safe!
        if acc_data.shape[1] == 252:
            save('models/score/kn1_ua_bertscore', test_data[:3])
        elif acc_data.shape[1] == 384:
            save('models/score/kn1_uq_bertscore.npy', test_data[:3])
        elif acc_data.shape[1] == 271:
            save('models/score_ml/jobber_ua_bertscore.npy', test_data[:3])
        elif acc_data.shape[1] == 602:
            save('models/score_ml/jobber_uq_bertscore.npy', test_data[:3])
        else:
            save('models/unkown_dataset_scoring/' + datetime.now().strftime("%Y-%m-%d_ %H-%M-%S"))

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)


# pytorch-lightning module to fine-tune model on verification feedback   
class LitVerFineT5(pl.LightningModule):

    def __init__(self, batch_size, mode, model):
        super(LitVerFineT5, self).__init__()
        if model == "mt5":
            print("Using mt5 base Model")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        else:
            print("Using t5 base Model")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.mode = mode
        self.batch_size = batch_size
        data = dl.T5Dataset('preprocessed/'+ mode + '_train.npy')
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.test_data = dl.T5Dataset('preprocessed/'+ mode + "_UA.npy")
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=10,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        loss = self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()
        self.log('Train-Cross-Entropy-Loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):
        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                    [x['label'] for x in outputs]]
        pred = extract_pred(val_data[0])
        truth = [x.split(':', 1)[1] for x in val_data[1]]
        label_pred = extract_label(val_data[0])
        acc_data = np.array([val_data[2], label_pred])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']

        self.log('my_metric', (sacrebleu_score / 100 + rouge_score + meteor_score) / 3 * val_macro)

        self.log('sacreBleu', sacrebleu_score)
        self.log('val_macro', val_macro)
        self.log('ROUGE', rouge_score)
        self.log('METEOR', meteor_score)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, sacrebleu_score, rouge_score, meteor_score))

    def test_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def test_epoch_end(self, outputs):
        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        test_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                     [x['original'] for x in outputs],
                     [x['label'] for x in outputs]]
        pred = extract_pred(test_data[0])
        truth = [x.split(':', 1)[1] for x in test_data[1]]
        label_pred = extract_label(test_data[0])
        acc_data = np.array([test_data[3], label_pred])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']
        for i in range(1, 50):
            print("Prediction:", label_pred[i], pred[i])
            print("Truth:", test_data[3][i], truth[i])
            #pass


        self.log('bleu', sacrebleu_score)
        self.log('macro_f1', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        self.log('acc', val_acc)
        self.log('weighted', val_weighted)
        print("Performance:")
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, sacrebleu_score, rouge_score, meteor_score))
        # FIXME: Currently recognizes different test sets by length, not safe!
        if acc_data.shape[1] == 252:
            save('models/'+ self.mode + '/kn1_ua_bertscore', test_data[:3])
        elif acc_data.shape[1] == 384:
            save('models/'+ self.mode + '/kn1_uq_bertscore.npy', test_data[:3])
        elif acc_data.shape[1] == 271:
            save('models/'+ self.mode + '/jobber_ua_bertscore.npy', test_data[:3])
        elif acc_data.shape[1] == 602:
            save('models/'+ self.mode + '/jobber_uq_bertscore.npy', test_data[:3])
        else:
            save('models/unkown_dataset_ver/' + datetime.now().strftime("%Y-%m-%d_ %H-%M-%S"))

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)


# pytorch-lightning module to fine-tune model on verification feedback
class LitMultiT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitMultiT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.batch_size = batch_size
        data = dl.T5Dataset('preprocessed/wq_ver_kn1_train.npy')
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.test_data = dl.T5Dataset('preprocessed/wq_ver_kn1_ua.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=10,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):
        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                    [x['label'] for x in outputs]]
        pred = extract_pred(val_data[0])
        truth = [x.split(':', 1)[1] for x in val_data[1]]
        label_pred = extract_label(val_data[0])
        acc_data = np.array([val_data[2], label_pred])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']

        self.log('my_metric', (sacrebleu_score / 100 + rouge_score + meteor_score) / 3 * val_macro)

        self.log('bleu', sacrebleu_score)
        self.log('val_macro', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, sacrebleu_score, rouge_score, meteor_score))

    def test_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def test_epoch_end(self, outputs):
        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        test_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                     [x['original'] for x in outputs],
                     [x['label'] for x in outputs]]
        pred = extract_pred(test_data[0])
        truth = [x.split(':', 1)[1] for x in test_data[1]]
        label_pred = extract_label(test_data[0])
        acc_data = np.array([test_data[3], label_pred])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']

        self.log('bleu', sacrebleu_score)
        self.log('macro_f1', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        self.log('acc', val_acc)
        self.log('weighted', val_weighted)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, sacrebleu_score, rouge_score, meteor_score))
        # FIXME: Currently recognizes different test sets by length, not safe!
        if acc_data.shape[1] == 252:
            np.save('ver_kn1_ua_bertscore.npy', np.array(test_data[:3]), allow_pickle=True)
        else:
            np.save('ver_kn1_uq_bertscore.npy', np.array(test_data[:3]), allow_pickle=True)

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)

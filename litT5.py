import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
from utils import *
import dataloading as dl
import warnings
import datasets
import torch
import numpy as np

# set seeds
pl.seed_everything(73)
# just here for cleaner console output
warnings.filterwarnings("ignore")

# text quality metrics
sacrebleu = datasets.load_metric('sacrebleu')
rouge = datasets.load_metric('rouge')
meteor = datasets.load_metric('meteor')


# pytorch-lightning module to fine-tune model on scores
class LitScoreFineT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitScoreFineT5, self).__init__()
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.batch_size = batch_size
        # Load and split data
        data = dl.T5Dataset('preprocessed/score_kn1_train.npy')
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.test_data = dl.T5Dataset('preprocessed/score_kn1_ua.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        # force min length of prediction
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=11,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        loss = self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()
        self.log('Cross-Entropy-Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
        truth = [x.split(' ', 2)[2] for x in val_data[1]]

        # calculate model selection metrics
        acc_data = np.array(val_data[2:])
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
        print('MSE = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(mse_val, sacrebleu_score, rouge_score, meteor_score))

        # FIXME: Currently recognizes different test sets by length, not safe!
        if acc_data.shape[1] == 252:
            save('models/score/kn1_ua_bertscore', test_data[:3])
        else:
            save('models/score/kn1_uq_bertscore.npy', test_data[:3])

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

    def __init__(self, batch_size, model=None):
        super(LitVerFineT5, self).__init__()

        if model != None:
            self.model = model
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.batch_size = batch_size
        data = dl.T5Dataset('preprocessed/ver_kn1_train.npy')
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.test_data = dl.T5Dataset('preprocessed/ver_kn1_ua.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=10,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        loss = self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()
        self.log('Cross-Entropy-Loss', loss, on_epoch=True)
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
            save('models/ver/kn1_ua_bertscore.npy', test_data[:3])
        else:
            save('models/ver/kn1_uq_bertscore.npy', test_data[:3])

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
        # KN1 dataset
        self.batch_size = batch_size
        kn1_normal = dl.T5Dataset('preprocessed/ver_kn1_train.npy')
        kn1_label = dl.T5Dataset('preprocessed/label_only_kn1.npy')
        self.kn1_train = random_split(kn1_label, split(len(kn1_label)),
                                                      generator=torch.Generator().manual_seed(42))[0]
        self.val_data = random_split(kn1_normal, split(len(kn1_normal)),
                                            generator=torch.Generator().manual_seed(42))[1]
        self.seb = dl.T5Dataset("preprocessed/seb_train.npy")
        self.cose = dl.T5Dataset('preprocessed/cose_train.npy')
        self.esnli = dl.T5Dataset('preprocessed/esnli_train.npy')
        self.test_data = dl.T5Dataset('preprocessed/ver_kn1_ua.npy')
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
        train_length = len(self.kn1_train)
        train_set = ConcatDataset(
            [
                get_subset(self.esnli, train_length),
                get_subset(self.cose, train_length),
                get_subset(self.seb, train_length),
                self.kn1_train
            ]
        )
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)

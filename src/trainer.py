from .models import CNNModel, BreastCancerModel

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data.sampler import SequentialSampler, RandomSampler, WeightedRandomSampler

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from sklearn.metrics import f1_score

class pl_Wrapper(pl.LightningModule):
    def __init__(self, args):
        super(pl_Wrapper, self).__init__()

        self.config = args
        # self.model = CNNModel(args)

        self.model = BreastCancerModel(args)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, imgs, cat_features, num_features):
        output = self.model(imgs, cat_features, num_features)
        return output

    def train_dataloader(self):
        loader = DataLoader(
                            self.config.train_dataset,
                            batch_size=self.config.train_params.batch_size,
                            num_workers=0,
                            shuffle=False,
                            sampler=RandomSampler(self.config.train_dataset),
                            # sampler=WeightedRandomSampler(self.config.samples_weight, len(self.config.samples_weight)),
                            drop_last=False,
                            pin_memory=True,
                        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
                            self.config.valid_dataset,
                            batch_size=self.config.train_params.val_batch_size,
                            num_workers=0,
                            shuffle=False,
                            sampler=SequentialSampler(self.config.valid_dataset),
                            drop_last=False,
                            pin_memory=True,
                        )
        return loader
    
    def sharing_step(self, batch):
        pred = self.forward(batch['img'], batch['cat_features'], batch['num_features'])
        
        loss = self.criterion(pred, batch['label'])
        
        # sigmoid
        pred = torch.sigmoid(pred)
        
        # softmax
        # pred = torch.softmax(pred, dim=1)
        
        return pred, loss

    def training_step(self, train_batch, batch_idx):
        pred, loss = self.sharing_step(train_batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {'loss':loss, 'pred':pred.clone().detach().cpu(), 'label':train_batch['label'].clone().detach().cpu()}
        # return {'loss':loss, 'pred':pred.tolist(), 'label':train_batch['label'].tolist()}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        preds = torch.cat([x['pred'] for x in outputs]).numpy()
        labels = torch.cat([x['label'] for x in outputs]).numpy().reshape(-1)
        
        # preds = np.argmax(preds, 1)
        preds = np.round(preds[:, 1])
        f1 = f1_score(labels, preds, average='macro')
        
        self.log("total_train_loss", avg_loss, logger=True)
        self.log("total_train_f1_score", f1, logger=True)

    def validation_step(self, val_batch, batch_idx):
        pred, loss = self.sharing_step(val_batch)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {'loss':loss, 'pred':pred.clone().detach().cpu(), 'label':val_batch['label'].clone().detach().cpu()}
        # return {'loss':loss, 'pred':pred.tolist(), 'label':val_batch['label'].tolist()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        preds = torch.cat([x['pred'] for x in outputs]).numpy()
        labels = torch.cat([x['label'] for x in outputs]).numpy().reshape(-1)
        
        # preds = np.argmax(preds, 1)
        preds = np.round(preds[:, 1])
        f1 = f1_score(labels, preds, average='macro')
        
        self.log("total_val_loss", avg_loss, logger=True)
        self.log("total_val_f1_score", f1, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.train_params.init_lr, weight_decay=self.config.train_params.weight_decay)
        # optimizer = Adam(self.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY)
        
        # num_train_steps = int(len(self.train)/(self.config.BATCH_SIZE*3)*self.config.EPOCHS)
        num_train_steps = int(len(self.train_dataloader()) * self.config.train_params.epochs)
        # num_train_steps = int(len(self.train_dataloader()) * 30)
        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                                             num_warmup_steps=int(num_train_steps*0.1), 
        #                                             num_training_steps=num_train_steps)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=int(num_train_steps*0.1), 
                                                                       num_training_steps=num_train_steps,
                                                                       num_cycles=self.config.train_params.sch_cycle)
        # scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=1)
        # scheduler_cosine = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)
        # scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine)
        
        return [optimizer], [{'scheduler':scheduler, 'interval':'step'}]